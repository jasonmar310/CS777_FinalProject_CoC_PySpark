# CS777_FinalProject_CoC_PySpark

In this project, we delve into the world of Clash of Clans to analyze and predict clan performances using PySpark.
Table of Contents


## Introduction

The dynamics and strategy within mobile games offer a comprehensive and complex
view of player behavior, team interactions, and performance metrics. To understand these
components and to gain insights into player preferences and clan dynamics, access to rich
datasets becomes pivotal. Also, the goal of this analysis could be instrumental in shaping
game event strategies. By understanding the dynamics of clan performance and the inherent
traits of high-performing clans, game developers can design targeted events that cater to
specific clan types. To sum up, we hope the results could help us have a bigger picture and
understanding of the game industry and learn about game activity planning strategies and
dynamics of user experience.

Research Question:
1. What traits distinguish high-performing clans in terms of member levels and trophies?
2. How does the type of clan (open, closed, etc.) impact its performance metrics?
3. Can we predict a clan's type based on its performance metrics and member statistics?

Data Source:
https://www.kaggle.com/datasets/asaniczka/clash-of-clans-clans-dataset-2023-3-5m-clans/data

-----

Getting Started

    Using Google Cloud Platform (GCP)

    Using this platform to run the Pyspark code and observe the output. In
    Dataproc, the cluster node configuration would recommend:
    Manager node: N2 (n2-standard-4/ 16GB
    Worker node: E2 (e2-standard-8/ 32GB

    The CoC data also has the link on Cloud Storage:
    gs://cs-777-2023/coc_clans_dataset.csv
    
    For project (main file):
    gs://cs-777-2023/project.py

    project2 (k-fold validation with same model):
    gs://cs-777-2023/project2.py

    project_plt (generate plot in missing value part):
    gs://cs-777-2023/project_plt.py
    
    Run the .py file with argument as CoC dataset.

