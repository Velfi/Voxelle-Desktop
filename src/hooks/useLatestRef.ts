import { useRef } from "react";

/**
 * Returns a ref that always holds the latest value.
 * Useful for accessing mutable state inside callbacks/effects
 * without adding it to dependency arrays.
 */
export function useLatestRef<T>(value: T): React.MutableRefObject<T> {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}
