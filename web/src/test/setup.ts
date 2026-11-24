// Vitest setup for the component tests. Globals are off (tests import from 'vitest'), so Testing
// Library cannot register its own cleanup; unmount every render here instead.

import { cleanup } from '@testing-library/react';
import { afterEach } from 'vitest';

afterEach(() => {
  cleanup();
});
