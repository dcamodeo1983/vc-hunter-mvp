
from collections import defaultdict

def compute_vc_relationships(enriched_data):
    # Reverse index: company -> list of VCs
    company_to_vcs = defaultdict(list)
    for vc, companies in enriched_data.items():
        for c in companies:
            company_to_vcs[c].append(vc)

    # Co-investment and competition scores
    relationships = defaultdict(lambda: {"collaboration": 0, "competition": 0})

    vcs = list(enriched_data.keys())
    for i, vc_a in enumerate(vcs):
        for j, vc_b in enumerate(vcs):
            if i >= j:
                continue
            shared = 0
            competitors = 0
            for company, vcs_list in company_to_vcs.items():
                if vc_a in vcs_list and vc_b in vcs_list:
                    shared += 1
                elif vc_a in vcs_list or vc_b in vcs_list:
                    competitors += 1
            key = tuple(sorted([vc_a, vc_b]))
            relationships[key]["collaboration"] = shared
            relationships[key]["competition"] = competitors

    return {f"{a} <-> {b}": vals for (a, b), vals in relationships.items()}
