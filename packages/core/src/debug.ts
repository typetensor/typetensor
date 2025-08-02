// Utility from https://stackoverflow.com/a/57683652
export type Expand<T> = T extends infer O ? { [K in keyof O]: O[K] } : never;

// Utility from https://stackoverflow.com/a/57683652
export type ExpandRecursively<T> = T extends object
  ? T extends infer O
    ? { [K in keyof O]: ExpandRecursively<O[K]> }
    : never
  : T;
