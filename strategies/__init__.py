from strategies.business_process import BusinessProcess
from strategies.requirements import Requirements
from strategies.generic_knowledge import GenericKnowledge


STRATEGY_REGISTRY = {
    "business_process": BusinessProcess(),
    "requirements": Requirements(),
    "generic_knowledge": GenericKnowledge(),
}


def get_strategy(document_type):
    """Return a strategy; unsupported categories use generic knowledge."""
    return STRATEGY_REGISTRY.get(
        document_type,
        STRATEGY_REGISTRY["generic_knowledge"],
    )
