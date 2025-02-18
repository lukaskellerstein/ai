import { makeStyles, shorthands } from "@griffel/react";

export const useClasses = makeStyles({
  app: { display: "flex", height: "100%", ...shorthands.overflow("hidden") },
});
