# Teams

## Teams architecture

`RoundRobinGroupChat`
`SelectorGroupChat`
`Swarm`

## Terminations

`MaxMessageTermination`: Stops after a specified number of messages have been produced, including both agent and task messages.

`TextMentionTermination`: Stops when specific text or string is mentioned in a message (e.g., “TERMINATE”).

`TokenUsageTermination`: Stops when a certain number of prompt or completion tokens are used. This requires the agents to report token usage in their messages.

`TimeoutTermination`: Stops after a specified duration in seconds.

`HandoffTermination`: Stops when a handoff to a specific target is requested. Handoff messages can be used to build patterns such as Swarm. This is useful when you want to pause the run and allow application or user to provide input when an agent hands off to them.

`SourceMatchTermination`: Stops after a specific agent responds.

`ExternalTermination`: Enables programmatic control of termination from outside the run. This is useful for UI integration (e.g., “Stop” buttons in chat interfaces).

`StopMessageTermination`: Stops when a StopMessage is produced by an agent.
