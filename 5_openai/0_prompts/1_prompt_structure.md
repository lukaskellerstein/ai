# Elements of a Prompt

As we cover more and more examples and applications with prompt engineering, you will notice that certain elements make up a prompt.

A prompt contains any of the following elements:

## Role

Role that AI should behave as.

```
You are a genious marketing director, that is very creative and all his ideas become a real success. People love your ideas.
```

## Context

- external information or additional context that can steer the model to better responses

Examples:

```
Here is an example of recipe that I really like.
===
<receipt>
===
```

## Output Indicator

- the type or format of the output.

You do not need all the four elements for a prompt and the format depends on the task at hand. We will touch on more concrete examples in upcoming guides.

Examples:

`Format the response, so I can create a landing page from it.`
`Write the response as a 5 year old.`

## Instruction

- a specific task or instruction you want the model to perform

Examples:

- "Write", "Classify", "Summarize", "Translate", "Order", etc.
