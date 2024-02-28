import { useClasses } from "./sidebar.styles";

export const SideBar = () => {
  const classes = useClasses();

  return (
    <div className={classes.sidebar}>
      <div className={classes.sidebarItem}>Chat 1</div>
      <div className={classes.sidebarItem}>Chat 2</div>
    </div>
  );
};
