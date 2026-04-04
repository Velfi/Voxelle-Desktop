/**
 * Per-generator custom hooks — barrel export.
 *
 * Each hook encapsulates the useState/useRef/useEffect/useStrokePhase cluster
 * for a single generator, keeping App.tsx focused on orchestration.
 */

export { useRocksGenerator, type RocksGeneratorState } from "./useRocksGenerator";
export { useGrassGenerator, type GrassGeneratorState } from "./useGrassGenerator";
export { useAshlarGenerator, type AshlarGeneratorState } from "./useAshlarGenerator";
export { useFloraGenerator, type FloraGeneratorState } from "./useFloraGenerator";
export { usePiscinaGenerator, type PiscinaGeneratorState } from "./usePiscinaGenerator";
export { useInsectaGenerator, type InsectaGeneratorState } from "./useInsectaGenerator";
export { useFaunaGenerator, type FaunaGeneratorState } from "./useFaunaGenerator";
export { useRopeClothGenerator, type RopeClothGeneratorState } from "./useRopeClothGenerator";
export { useRoofGenerator, type RoofGeneratorState } from "./useRoofGenerator";
