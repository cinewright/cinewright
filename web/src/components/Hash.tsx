import { fullHash, shortHash } from '../lib/format';
import { CopyValue } from './CopyValue';

/** SHA-256 truncated in the middle, `sha256:<hex>` in the tooltip and through the copy button. */
export function Hash({ value, label }: { value: string; label: string }) {
  return <CopyValue text={shortHash(value)} value={fullHash(value)} label={label} className="text-hash hash__value" />;
}
