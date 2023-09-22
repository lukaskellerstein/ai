import { makeStyles, shorthands, tokens } from "@fluentui/react-components";

export const useClasses = makeStyles({
  sidebar: {
    height: "100%",
    width: "75px",
    maxWidth: "150px",
    ...shorthands.padding(0, "10px"),
    ...shorthands.borderRight("1px", "solid", tokens.colorNeutralForeground2),
  },
  sidebarItem: { marginTop: "10px" },
});
