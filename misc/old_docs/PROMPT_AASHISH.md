# Chief Integration Engineer Contract

task:
  name: "Drift-Sense++ Final Integration"
  owner: "Aashish"
  branch: "main"
  role: "Chief Integration Engineer"

mission:
  objective: >
    Produce the final competition-ready Drift-Sense++ inference system
    that satisfies the Applied Materials repository contract and
    maximizes localization score without sacrificing reproducibility.

absolute_ownership:
  - inference.py
  - dataset_generator.py
  - README.md
  - requirements.txt
  - production_engine/
  - submission_package/
  - final benchmark
  - final architecture
  - main branch

teammates:
  Sai:
    component: retrieval
  Akhilesh:
    component: ranking
  Shanganidhi:
    component: confidence

integration_rule:
  never:
    - blindly merge teammate code
    - trust reported metrics without reproduction
    - modify benchmark to improve score
    - copy entire teammate branch into main
  always:
    - inspect diff
    - reproduce experiment
    - compare against baseline
    - run frozen benchmark
    - perform regression test
    - record KEEP/MODIFY/REJECT

final_pipeline:
  - input_validation
  - reference_search_preprocessing
  - candidate_retrieval
  - candidate_rescue
  - candidate_ranking
  - pose_refinement
  - confidence_evaluation
  - final_coordinate

acceptance_gate:
  retrieval:
    require: "measurable improvement or justified KEEP"
  ranking:
    require: "measurable conditional Top-1 improvement"
  confidence:
    require: "improved selective performance without unacceptable coverage loss"
  latency:
    require: "within competition constraints"
  reproducibility:
    require: "fresh-machine execution"

competition_contract:
  inference:
    inputs:
      - reference_image_path
      - search_image_path
    output:
      - x
      - y
  generator:
    inputs:
      - architecture_style
      - number_of_pairs
      - output_directory
    output:
      - image_pairs
      - ground_truth_coordinates

final_rule: >
  Main is the only branch that becomes the submission. Teammate branches
  are research laboratories. Nothing enters main without independent
  verification.
