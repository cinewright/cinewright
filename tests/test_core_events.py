"""Event log: live broadcast across threads, replay after Last-Event-ID, end of job-filtered streams, kept previews."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from conftest import finish_job, leased_job, preview_event, report

from cinewright.api.schemas import ClientKind, JobFailure, JobRequest, JobState, WorkerRegistration
from cinewright.core.clock import to_iso
from cinewright.core.errors import Invalid

REQUEST = JobRequest.model_validate({"capability": "t2i", "prompt": "a lighthouse"})
PROGRESS = {"kind": "progress", "data": {"phase": "inference", "step": 1, "total_steps": 4, "percent": 25}}
LOG = {"kind": "log", "data": {"message": "step done"}}


@pytest.fixture(autouse=True)
def worker(core):
    registration = WorkerRegistration(host_id="local", engine="wan2gp", models=["test_image"], health={"state": "ready"})
    return core.register_worker(registration, core.enrollment_token).worker_id


def collect(core, job_id=None, after=None, count=None):
    async def run():
        events = []
        async for event in core.events(job_id, after):
            events.append(event)
            if count is not None and len(events) == count:
                break
        return events

    return asyncio.run(asyncio.wait_for(run(), 5))


def collect_for(core, job_id=None, after=None, seconds=0.3):
    """Everything a stream delivers within `seconds`."""

    async def run():
        events = []

        async def follow():
            async for event in core.events(job_id, after):
                events.append(event)

        try:
            await asyncio.wait_for(follow(), seconds)
        except TimeoutError:
            pass
        return events

    return asyncio.run(run())


def test_live_events_from_other_threads_reach_the_stream(core):
    async def scenario():
        received = []

        async def follow():
            async for event in core.events(None, None):
                received.append(event)
                if len(received) == 2:
                    return

        task = asyncio.create_task(follow())
        await asyncio.sleep(0.05)
        first = await asyncio.to_thread(core.submit_job, REQUEST, ClientKind.UI, None)
        await asyncio.to_thread(core.cancel_job, first.id)
        await asyncio.wait_for(task, 5)
        return first.id, received

    job_id, received = asyncio.run(scenario())
    assert [(e.job_id, e.seq, e.data["state"]) for e in received] == [(job_id, 1, "queued"), (job_id, 2, "cancelled")]


def test_job_stream_replays_history_and_ends_after_the_terminal_state(core):
    job_id = core.submit_job(REQUEST, ClientKind.UI, None).id
    core.cancel_job(job_id)
    assert [e.seq for e in collect(core, job_id)] == [1, 2]
    assert [e.seq for e in collect(core, job_id, after=f"{job_id}:1")] == [2]
    assert collect(core, job_id, after=f"{job_id}:2") == []
    assert collect(core, "no-such-job") == []


def test_unfiltered_stream_replays_after_last_event_id(core):
    first = core.submit_job(REQUEST, ClientKind.UI, None).id
    second = core.submit_job(REQUEST, ClientKind.UI, None).id
    core.cancel_job(first)
    events = collect(core, None, after=f"{first}:1", count=2)
    assert [(e.job_id, e.seq) for e in events] == [(second, 1), (first, 2)]


def test_malformed_last_event_id_is_ignored(core):
    job_id = core.submit_job(REQUEST, ClientKind.UI, None).id
    core.cancel_job(job_id)
    assert [e.seq for e in collect(core, job_id, after="garbage")] == [1, 2]


def test_closing_the_core_ends_open_streams(core):
    async def scenario():
        task = asyncio.create_task(_drain(core))
        await asyncio.sleep(0.05)
        await asyncio.to_thread(core.close)
        return await asyncio.wait_for(task, 5)

    assert asyncio.run(scenario()) == []


async def _drain(core):
    return [event async for event in core.events(None, None)]


def test_a_resumed_stream_replays_no_previews_then_sends_the_kept_latest_one(core):
    worker, job_id, lease_id = leased_job(core)  # seq 1 queued, 2 leased
    report(core, worker, job_id, lease_id, PROGRESS, preview_event(1))  # 3 running, 4 progress
    report(core, worker, job_id, lease_id, preview_event(2), LOG, preview_event(3))  # 5 log
    assert core.db.query_one("SELECT COUNT(*) FROM job_events WHERE kind = 'preview'")[0] == 0
    for stream_filter in (job_id, None):
        events = collect_for(core, stream_filter, after=f"{job_id}:3")
        assert [(e.job_id, e.seq, e.kind) for e in events] == [(job_id, 4, "progress"), (job_id, 5, "log"), (job_id, 5, "preview")]
        assert events[-1].data == preview_event(3)["data"]


def test_preview_rows_stored_by_an_older_release_are_never_replayed(core, clock):
    worker, job_id, lease_id = leased_job(core)  # seq 1 queued, 2 leased
    report(core, worker, job_id, lease_id, PROGRESS)  # 3 running, 4 progress
    legacy = {"media_type": "image/png", "data_base64": "bGVnYWN5"}
    with core.db.transaction() as conn:  # releases 0.0.1 and 0.0.2 stored every preview in job_events
        conn.execute(
            "INSERT INTO job_events (job_id, seq, kind, data, created_at) VALUES (?, 5, 'preview', ?, ?)",
            (job_id, json.dumps(legacy), to_iso(clock())),
        )
    report(core, worker, job_id, lease_id, LOG, preview_event(1))  # 6 log: the next seq after the old row
    kept = (6, "preview")  # the live preview kept in memory, with the seq of the latest stored event
    cases = {  # name: (job filter, Last-Event-ID, expected (seq, kind))
        "fresh job-filtered": (job_id, None, [(1, "state"), (2, "state"), (3, "state"), (4, "progress"), (6, "log"), kept]),
        "fresh unfiltered": (None, None, [kept]),
        "job-filtered resumed": (job_id, f"{job_id}:3", [(4, "progress"), (6, "log"), kept]),
        "unfiltered resumed": (None, f"{job_id}:3", [(4, "progress"), (6, "log"), kept]),
        # 0.0.1 and 0.0.2 sent previews with an `id:`, so a browser can resume from the old row itself.
        "job-filtered resumed after the old row": (job_id, f"{job_id}:5", [(6, "log"), kept]),
        "unfiltered resumed after the old row": (None, f"{job_id}:5", [(6, "log"), kept]),
    }
    for name, (stream_filter, after, expected) in cases.items():
        events = collect_for(core, stream_filter, after)
        assert [(e.seq, e.kind) for e in events] == expected, name
        assert events[-1].data == preview_event(1)["data"], name  # the kept live preview, not the old row


@pytest.mark.parametrize("filtered", [True, False], ids=["job-filtered", "unfiltered"])
def test_a_preview_committed_between_subscribe_and_snapshot_arrives_once(core, monkeypatch, filtered):
    worker, job_id, lease_id = leased_job(core)  # seq 1 queued, 2 leased
    replay = core.event_log._replay
    calls = []

    def replay_after_a_report(*args):
        # The stream has subscribed, so this commit is queued for it and also seen by the snapshot.
        if not calls:
            with ThreadPoolExecutor(1) as pool:
                pool.submit(report, core, worker, job_id, lease_id, LOG, preview_event(2)).result()  # 3 running, 4 log
        calls.append(args)
        return replay(*args)

    monkeypatch.setattr(core.event_log, "_replay", replay_after_a_report)
    events = collect_for(core, job_id if filtered else None, after=None if filtered else f"{job_id}:2")
    assert calls, "the stream did not take a replay snapshot"
    expected = [(1, "state"), (2, "state")] if filtered else []
    assert [(e.seq, e.kind) for e in events] == [*expected, (3, "state"), (4, "log"), (4, "preview")]
    assert events[-1].data == preview_event(2)["data"]


def test_a_new_stream_sends_the_kept_preview_of_each_running_job_it_covers(core):
    first, second = leased_job(core), leased_job(core)
    report(core, *first, preview_event(1))
    report(core, *second, preview_event(2))
    report(core, *first, preview_event(3))
    unfiltered = collect_for(core)  # no replay without Last-Event-ID, only the kept previews
    assert sorted((e.job_id, e.data["data_base64"]) for e in unfiltered) == sorted(
        [(first[1], preview_event(3)["data"]["data_base64"]), (second[1], preview_event(2)["data"]["data_base64"])]
    )
    filtered = collect_for(core, first[1])  # the whole history, then the job's kept preview
    assert [(e.seq, e.kind) for e in filtered] == [(1, "state"), (2, "state"), (3, "state"), (3, "preview")]
    assert filtered[-1].data == preview_event(3)["data"]


def test_no_preview_is_kept_once_the_job_finishes(core):
    worker, job_id, lease_id = leased_job(core)
    report(core, worker, job_id, lease_id, preview_event(1))
    assert [e.data for e in core.event_log.kept_previews(job_id)] == [preview_event(1)["data"]]
    finish_job(core, worker, job_id, lease_id)
    assert core.event_log.kept_previews(job_id) == []
    assert core.event_log.kept_previews() == []
    assert [e.kind for e in collect_for(core, job_id)].count("preview") == 0


def test_a_rejected_batch_keeps_no_preview_and_a_requeue_drops_the_kept_one(core):
    worker, job_id, lease_id = leased_job(core)
    with pytest.raises(Invalid):
        report(core, worker, job_id, lease_id, preview_event(1), {"kind": "progress", "data": {"percent": 500}})
    assert core.event_log.kept_previews(job_id) == []
    report(core, worker, job_id, lease_id, preview_event(2))
    failure = JobFailure(lease_id=lease_id, error={"code": "oom", "message": "The GPU ran out of memory.", "retryable": True})
    core.fail_job(worker, job_id, failure)
    assert core.get_job(job_id).state == JobState.QUEUED
    assert core.event_log.kept_previews(job_id) == []
