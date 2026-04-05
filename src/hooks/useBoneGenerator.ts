import { useState, useRef, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useStrokePhase, type StrokePhaseHandle } from "../useStrokePhase";

interface BoneGeneratorContext {
  activeColorRef: React.MutableRefObject<number>;
  activeMaterialRef: React.MutableRefObject<string>;
}

export interface BoneGeneratorState {
  bonePhase: StrokePhaseHandle<Record<string, never>>;
  boneMode: "add" | "edit" | "delete";
  setBoneMode: (m: "add" | "edit" | "delete") => void;
  boneModeRef: React.RefObject<string>;
  boneJointCount: number;
  setBoneJointCount: React.Dispatch<React.SetStateAction<number>>;
  boneBoneCount: number;
  setBoneBoneCount: React.Dispatch<React.SetStateAction<number>>;
  boneDefaultRadius: number;
  setBoneDefaultRadius: React.Dispatch<React.SetStateAction<number>>;
  boneDefaultRadiusRef: React.RefObject<number>;
  ikEnabled: boolean;
  setIkEnabled: React.Dispatch<React.SetStateAction<boolean>>;
  ikEnabledRef: React.RefObject<boolean>;
}

export function useBoneGenerator(ctx: BoneGeneratorContext): BoneGeneratorState {
  const [boneMode, setBoneModeState] = useState<"add" | "edit" | "delete">("add");
  const boneModeRef = useRef<string>("add");
  const [boneJointCount, setBoneJointCount] = useState(0);
  const [boneBoneCount, setBoneBoneCount] = useState(0);
  const [boneDefaultRadius, setBoneDefaultRadius] = useState(3);
  const boneDefaultRadiusRef = useRef(3);
  const [ikEnabled, setIkEnabled] = useState(true);
  const ikEnabledRef = useRef(true);

  const setBoneMode = (m: "add" | "edit" | "delete") => {
    setBoneModeState(m);
    boneModeRef.current = m;
  };

  const bonePhase = useStrokePhase<Record<string, never>>({
    phases: ["build", "pose"],
    onCancel: () => {
      void invoke("bone_session_clear")
        .then(() => {
          setBoneJointCount(0);
          setBoneBoneCount(0);
        })
        .catch(() => {});
    },
    onCommit: () => {
      void invoke("bone_session_commit", {
        args: {
          color: ctx.activeColorRef.current,
          material: ctx.activeMaterialRef.current,
        },
      })
        .then(() => invoke("bone_session_clear"))
        .then(() => {
          setBoneJointCount(0);
          setBoneBoneCount(0);
        })
        .catch(() => {});
    },
  });

  // When the shared gizmo is dragged, sync the selected joint position.
  useEffect(() => {
    const unlisten = listen<[number, number, number]>("generator-gizmo-moved", (ev) => {
      const [x, y, z] = ev.payload;
      // Get the selected joint and update its position to match the gizmo.
      void invoke<any>("bone_session_get")
        .then((session: any) => {
          const sel = session?.selected;
          if (!sel) return;
          const jointId = sel.joint ?? (sel.bone != null
            ? session.bones?.find((b: any) => b.id === sel.bone)?.jointA
            : null);
          if (jointId == null) return;
          return invoke("bone_set_joint_position", {
            args: { id: jointId, x, y, z },
          });
        })
        .catch(() => {});
    });
    return () => { void unlisten.then((u) => u()); };
  }, []);

  // When the scale ring is dragged, sync the selected joint radius.
  useEffect(() => {
    const unlisten = listen<number>("generator-gizmo-scaled", (ev) => {
      const radius = ev.payload;
      void invoke<any>("bone_session_get")
        .then((session: any) => {
          const sel = session?.selected;
          if (!sel?.joint) return;
          return invoke("bone_set_joint_radius", {
            args: { id: sel.joint, radius },
          });
        })
        .catch(() => {});
    });
    return () => { void unlisten.then((u) => u()); };
  }, []);

  return {
    bonePhase,
    boneMode,
    setBoneMode,
    boneModeRef,
    boneJointCount,
    setBoneJointCount,
    boneBoneCount,
    setBoneBoneCount,
    boneDefaultRadius,
    setBoneDefaultRadius: (n: React.SetStateAction<number>) => {
      setBoneDefaultRadius(n);
      if (typeof n === "number") boneDefaultRadiusRef.current = n;
    },
    boneDefaultRadiusRef,
    ikEnabled,
    setIkEnabled: (v: React.SetStateAction<boolean>) => {
      setIkEnabled(v);
      if (typeof v === "boolean") ikEnabledRef.current = v;
    },
    ikEnabledRef,
  };
}
