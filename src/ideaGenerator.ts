import adjectivesRaw from "./adjectives.txt?raw";
import nounsRaw from "./nouns.txt?raw";

const ADJECTIVES = adjectivesRaw
  .split("\n")
  .map((s) => s.trim())
  .filter(Boolean);
const NOUNS = nounsRaw
  .split("\n")
  .map((s) => s.trim())
  .filter(Boolean);

/** Each template must contain exactly one {a} and one {n}. */
const TEMPLATES = [
  "I wish I had a {a} {n}.",
  "Has anyone seen my {a} {n}?",
  "I saw a {a} {n} today!",
  "Try making a {a} {n}!",
  "What if there was a {a} {n}?",
  "I dreamt about a {a} {n} last night.",
  "Nobody ever builds a {a} {n}.",
  "Make a {a} {n} and see what happens.",
  "The world needs more {a} {n}s.",
  "One day I will own a {a} {n}.",
  "Found: one {a} {n}. Please claim.",
  "Lost: my {a} {n}. Reward offered.",
  "A {a} {n} would solve all my problems.",
  "I can't stop thinking about a {a} {n}.",
  "Beware of the {a} {n}.",
  "There's a {a} {n} behind you.",
  "For sale: slightly used {a} {n}.",
  "Breaking news: {a} {n} found locally.",
  "Today's forecast: {a} {n} incoming.",
  "Scientists discover a {a} {n}.",
  "Local resident builds {a} {n}. Neighbours stunned.",
  "Step 1: build a {a} {n}. Step 2: ???",
  "I have too many {a} {n}s.",
  "Tell me about your {a} {n}.",
  "This used to be a {a} {n}.",
  "Warning: do not feed the {a} {n}.",
  "The {a} {n} has escaped again.",
  "My therapist says I should build a {a} {n}.",
  "They said a {a} {n} couldn't be done.",
  "A {a} {n} a day keeps the doctor away.",
  "To-do list: buy milk, build {a} {n}.",
  "Accepting donations for my {a} {n} fund.",
  "Do not touch the {a} {n}.",
  "Legend speaks of a {a} {n}.",
  "I regret not buying that {a} {n} when I had the chance.",
  "My ex took the {a} {n} in the divorce.",
  "Haunted by visions of a {a} {n}.",
  "Plot twist: it was a {a} {n} all along.",
  "Instructions unclear. Built a {a} {n}.",
  "Fun fact: no one has ever made a {a} {n}. Until now.",
  "This is fine. The {a} {n} is fine.",
  "Error 404: {a} {n} not found.",
  "A {a} {n} walks into a bar.",
  "New year, new {a} {n}.",
  "Sponsored by: the {a} {n} foundation.",
  "Help I accidentally made a {a} {n}.",
  "The ancient prophecy spoke of a {a} {n}.",
  "If you build a {a} {n}, they will come.",
  "I come bearing gifts. The gift is a {a} {n}.",
  "Phase 1: {a} {n}. Phase 2: world domination.",
];

function pick<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

export function generateIdea(): string {
  const adj = pick(ADJECTIVES);
  const noun = pick(NOUNS);
  return pick(TEMPLATES).replace("{a}", adj).replace("{n}", noun);
}
