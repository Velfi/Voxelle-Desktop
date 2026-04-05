import { useState, useRef } from "react";
import { useLatestRef } from "./useLatestRef";

export interface RoofGeneratorState {
  roofStyle: string;
  setRoofStyle: React.Dispatch<React.SetStateAction<string>>;
  roofStyleRef: React.MutableRefObject<string>;
  roofHeight: number;
  setRoofHeight: React.Dispatch<React.SetStateAction<number>>;
  roofHeightRef: React.MutableRefObject<number>;
  roofHollow: boolean;
  setRoofHollow: React.Dispatch<React.SetStateAction<boolean>>;
  roofHollowRef: React.MutableRefObject<boolean>;
  roofPins: [number, number, number][];
  setRoofPins: React.Dispatch<React.SetStateAction<[number, number, number][]>>;
  roofPinsRef: React.MutableRefObject<[number, number, number][]>;
  roofAreaShape: "polygon" | "square" | "circle";
  setRoofAreaShape: React.Dispatch<React.SetStateAction<"polygon" | "square" | "circle">>;
  roofAreaShapeRef: React.MutableRefObject<"polygon" | "square" | "circle">;
  roofFirstClick: [number, number, number] | null;
  setRoofFirstClick: React.Dispatch<React.SetStateAction<[number, number, number] | null>>;
  roofFirstClickRef: React.MutableRefObject<[number, number, number] | null>;
}

export function useRoofGenerator(): RoofGeneratorState {
  const [roofStyle, setRoofStyle] = useState<string>("gable");
  const [roofHeight, setRoofHeight] = useState(6);
  const [roofHollow, setRoofHollow] = useState(false);
  const [roofPins, setRoofPins] = useState<[number, number, number][]>([]);
  const [roofAreaShape, setRoofAreaShape] = useState<"polygon" | "square" | "circle">("polygon");
  const [roofFirstClick, setRoofFirstClick] = useState<[number, number, number] | null>(null);

  const roofStyleRef = useLatestRef(roofStyle);
  const roofHeightRef = useLatestRef(roofHeight);
  const roofHollowRef = useLatestRef(roofHollow);
  // roofPins and roofFirstClick are mutated directly in handlers for immediate consistency.
  const roofPinsRef = useRef<[number, number, number][]>([]);
  const roofAreaShapeRef = useLatestRef(roofAreaShape);
  const roofFirstClickRef = useRef<[number, number, number] | null>(null);

  return {
    roofStyle,
    setRoofStyle,
    roofStyleRef,
    roofHeight,
    setRoofHeight,
    roofHeightRef,
    roofHollow,
    setRoofHollow,
    roofHollowRef,
    roofPins,
    setRoofPins,
    roofPinsRef,
    roofAreaShape,
    setRoofAreaShape,
    roofAreaShapeRef,
    roofFirstClick,
    setRoofFirstClick,
    roofFirstClickRef,
  };
}
