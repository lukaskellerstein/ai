import graphrag.api as api
from graphrag.index.typing import PipelineRunResult
import yaml
from graphrag.config.create_graphrag_config import create_graphrag_config
import asyncio
import pandas as pd
from pprint import pprint

PROJECT_DIRECTORY = "."  # noqa: PTH123
settings = yaml.safe_load(open(f"{PROJECT_DIRECTORY}/settings.yaml"))  # noqa: PTH123, SIM115
graphrag_config = create_graphrag_config(values=settings, root_dir=PROJECT_DIRECTORY)

async def main() -> None:
    # --------------------------
    # Build and index
    # --------------------------
    index_result: list[PipelineRunResult] = await api.build_index(config=graphrag_config)

    # index_result is a list of workflows that make up the indexing pipeline that was run
    for workflow_result in index_result:
        status = f"error\n{workflow_result.errors}" if workflow_result.errors else "success"
        print(f"Workflow Name: {workflow_result.workflow}\tStatus: {status}")


    # --------------------------
    # Query index
    # --------------------------
    final_entities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/entities.parquet")
    final_communities = pd.read_parquet(f"{PROJECT_DIRECTORY}/output/communities.parquet")
    final_community_reports = pd.read_parquet(
        f"{PROJECT_DIRECTORY}/output/community_reports.parquet"
    )
    response, context = await api.global_search(
        config=graphrag_config,
        entities=final_entities,
        communities=final_communities,
        community_reports=final_community_reports,
        community_level=2,
        dynamic_community_selection=False,
        response_type="Multiple Paragraphs",
        query="Jaké investiční strategie jsou porovnávány mezi sebou?",
    )

    print(response)

    pprint(context)  # noqa: T203




asyncio.run(main())