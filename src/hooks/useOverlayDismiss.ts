import React, { useRef } from "react";

export function useOverlayDismiss(onDismiss: () => void) {
  const pointerDownOnOverlay = useRef(false);

  return {
    onPointerDown: (e: React.PointerEvent<HTMLDivElement>) => {
      pointerDownOnOverlay.current = e.target === e.currentTarget;
    },
    onClick: (e: React.MouseEvent<HTMLDivElement>) => {
      if (e.target === e.currentTarget && pointerDownOnOverlay.current) {
        onDismiss();
      }
      pointerDownOnOverlay.current = false;
    },
  };
}
