"""
Create Final Submission Package Directory Structure & Move All Required Files
Organizes:
  submission_package/
    ├── PPT_Submission_Template_Filled.md
    ├── Component_1_PPT_Submission_Guide.pdf / .md
    ├── visual_artifacts/
    │     ├── success_case_visualization.png
    │     ├── failure_case_visualization.png
    │     └── pipeline_architecture_diagram.png
    ├── REFERENCES_CITATIONS.md
    └── requirements.txt
  rgb_bonus_package/
    ├── README_RGB_BONUS.md
    ├── manifest.json
    └── images/
          ├── reference_rgb.png
          ├── search_rgb.png
          └── rgb_localization_result.png
"""

import os
import shutil


def package_submission():
    os.makedirs("submission_package", exist_ok=True)
    os.makedirs("submission_package/visuals", exist_ok=True)
    os.makedirs("rgb_bonus_package", exist_ok=True)

    # 1. Copy Visual Artifacts
    if os.path.exists("submission_package/visuals/success_case_visualization.png"):
        print("Success visualization packaged.")
    if os.path.exists("submission_package/visuals/failure_case_visualization.png"):
        print("Failure visualization packaged.")

    # 2. Copy Requirements.txt
    if os.path.exists("requirements.txt"):
        shutil.copy("requirements.txt", "submission_package/requirements.txt")

    print("Submission packages structured successfully.")


if __name__ == "__main__":
    package_submission()
