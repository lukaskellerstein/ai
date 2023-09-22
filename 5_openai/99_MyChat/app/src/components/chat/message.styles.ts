import { makeStyles, shorthands, tokens } from '@fluentui/react-components';
export const useMessageStyles = makeStyles({
    container: {
        width: "auto",
        maxWidth: "60%",
        ...shorthands.padding("10px"),
        ...shorthands.borderRadius("5px"),
        marginBottom: "10px",
    },
    systemMessage: {
        maxWidth: "100%",
        textAlign: "center",
        backgroundColor: tokens.colorNeutralBackground4
    },
    assistantMessage: {
        alignSelf: "flex-start",
        backgroundColor: tokens.colorPaletteBlueBackground2
    },
    userMessage: {
        alignSelf: "flex-end",
        backgroundColor: tokens.colorPaletteForestBackground2
    },
});