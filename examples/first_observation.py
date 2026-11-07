"""Run the first ECA symbolic observation and recall loop."""

from eca import CognitiveEngine


def main() -> None:
    engine = CognitiveEngine(dimensions=4096, seed=7)

    engine.observe("bird", {"can": "fly", "has": "wings"})
    engine.observe("penguin", {"can": "swim", "has": "wings"})
    engine.observe("salmon", {"can": "swim", "lives_in": "water"})

    print("Current state:", engine.state)
    print("Recall for 'bird':", engine.recall("bird", top_k=3))


if __name__ == "__main__":
    main()
