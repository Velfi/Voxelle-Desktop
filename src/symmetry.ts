import type { MutableRefObject } from "react";

export function mirrorAxesFromFlags(mirrorX: boolean, mirrorY: boolean, mirrorZ: boolean): number {
  return (mirrorX ? 1 : 0) | (mirrorY ? 2 : 0) | (mirrorZ ? 4 : 0);
}

export function mirrorAxesFromRefs(
  mirrorXRef: MutableRefObject<boolean>,
  mirrorYRef: MutableRefObject<boolean>,
  mirrorZRef: MutableRefObject<boolean>,
): number {
  return mirrorAxesFromFlags(mirrorXRef.current, mirrorYRef.current, mirrorZRef.current);
}
