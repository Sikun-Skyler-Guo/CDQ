# data_loader.py
# Responsible for providing initial input data from CSV files

import pandas as pd
import ast  # Used to convert string representations of lists to actual lists

def load_initial_data(with_question=True, dataset_file="res_30_0929.csv"):
    """
    Load initial paper data (x0) and an optional question from CSV file.

    Args:
        with_question (bool): Whether to include the question.
        dataset_file (str): Path to CSV file containing paper data.

    Returns:
        dict: A dictionary containing 'papers' and 'question'.
    """
    try:
        # Load CSV file
        dataset_df = pd.read_csv(dataset_file)

        papers = []
        question = None

        # Iterate over unique targetPaperId
        for pid in dataset_df["targetPaperId"].unique():
            # Get the corresponding rows for this paper ID
            target_rows = dataset_df[dataset_df["targetPaperId"] == pid]
            if target_rows.empty:
                continue

            # Take title from the first row (all rows for same paper ID are identical)
            title = target_rows.iloc[0]["target_title"]

            # Parse reference abstracts
            ref_abs_str = target_rows.iloc[0].get("ref_abs", "[]")
            try:
                refs_abs = ast.literal_eval(ref_abs_str)
            except:
                refs_abs = []

            # Take questions from the first row if requested
            if with_question and "questions" in target_rows.columns:
                question_text = target_rows.iloc[0].get("questions", "[]")
                try:
                    questions = ast.literal_eval(question_text)
                except:
                    questions = []
            else:
                questions = None

            # Construct a single paper entry
            papers.append({
                "targetPaperId": pid,
                "title": title,
                "references_abs": refs_abs,
                "questions": questions
            })

        # Optionally, take a global question from the dataset
        if with_question and "questions" in dataset_df.columns:
            non_null_q = dataset_df["questions"].dropna()
            if not non_null_q.empty:
                question = non_null_q.iloc[0]

        return {"papers": papers, "question": question}

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return {"papers": [], "question": None}
    except Exception as e:
        print(f"Error while loading data: {e}")
        return {"papers": [], "question": None}


def load_eval_data(args, dataset_file="res_15.csv", with_question=True):
    """
    Load evaluation data from CSV file and prepare the paper ideas according to the indicator and type.

    Args:
        args: Parsed command-line arguments containing 'type' and 'indicator'.
        dataset_file (str): Path to CSV file containing evaluation data.
        with_question (bool): Whether to use 'question_' prefixed columns.

    Returns:
        dict: A dictionary containing 'papers'.
    """
    try:
        # Load CSV file
        dataset_df = pd.read_csv(dataset_file)

        papers = []

        # Iterate over unique targetPaperId
        for pid in dataset_df["targetPaperId"].unique():
            # Get the corresponding rows for this paper ID
            target_rows = dataset_df[dataset_df["targetPaperId"] == pid]
            if target_rows.empty:
                continue
            
            indicator = args.indicator
            type_ = args.type
            prefix = "question_" if with_question else ""  # Prefix columns if using question-based data

            # Take the first row for this paper
            row = target_rows.iloc[0]

            # Construct a complete record, placing ideas 1,2,3 into separate fields
            paper_record = {
                "targetPaperId": pid,
                "idea_a": row.get("target_paper_research_idea"),
                "idea_b": row.get(f"{prefix}{type_}_{indicator}_1"),
                "idea_c": row.get(f"{prefix}{type_}_{indicator}_2"),
                "idea_d": row.get(f"{prefix}{type_}_{indicator}_3")
            }

            papers.append(paper_record)
        
        return {"papers": papers}

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return {"papers": [], "question": None}
    except Exception as e:
        print(f"Error while loading data: {e}")
        return {"papers": [], "question": None}

def load_eval_data_winrate(args, dataset_file="res_30_0929_v7.csv"):
    """
    Load evaluation data from CSV file and prepare the paper ideas with and without 'question_' prefix.

    Args:
        args: Parsed command-line arguments containing 'type' and 'indicator'.
        dataset_file (str): Path to CSV file containing evaluation data.

    Returns:
        dict: A dictionary containing 'papers'.
    """
    try:
        # Load CSV file
        dataset_df = pd.read_csv(dataset_file)

        papers = []

        # Iterate over unique targetPaperId
        for pid in dataset_df["targetPaperId"].unique():
            # Get the corresponding rows for this paper ID
            target_rows = dataset_df[dataset_df["targetPaperId"] == pid]
            if target_rows.empty:
                continue

            indicator = args.indicator
            type_ = args.type

            # Take the first row for this paper
            row = target_rows.iloc[0]

            # Construct a record with both versions
            paper_record = {
                "targetPaperId": pid,
                "idea_b": row.get(f"{type_}_{indicator}_1"),
                "idea_c": row.get(f"{type_}_{indicator}_2"),
                "idea_d": row.get(f"{type_}_{indicator}_3"),
                "q_idea_b": row.get(f"question_{type_}_{indicator}_1"),
                "q_idea_c": row.get(f"question_{type_}_{indicator}_2"),
                "q_idea_d": row.get(f"question_{type_}_{indicator}_3")
            }

            papers.append(paper_record)

        return {"papers": papers}

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return {"papers": [], "question": None}
    except Exception as e:
        print(f"Error while loading data: {e}")
        return {"papers": [], "question": None}


def load_eval_data_winrate_max(args, dataset_file="res_30_1007_v1.csv"):
    try:
        # Load CSV file
        dataset_df = pd.read_csv(dataset_file)

        papers = []

        # Iterate over unique targetPaperId
        for pid in dataset_df["targetPaperId"].unique():
            # Get the corresponding rows for this paper ID
            target_rows = dataset_df[dataset_df["targetPaperId"] == pid]
            if target_rows.empty:
                continue

            indicator = args.indicator
            type_ = "rag"
            type_1 = "single"
            pre_ = "question_" if args.withQ else ""

            # Take the first row for this paper
            row = target_rows.iloc[0]

            # Construct a record with both versions
            paper_record = {
                "targetPaperId": pid,
                # Without 'question_' prefix
                "idea_a": row.get("target_paper_research_idea"),
                "idea_b": row.get(f"{pre_}{type_}_{indicator}_1"),
                "idea_c": row.get(f"{pre_}{type_}_{indicator}_2"),
                "idea_d": row.get(f"{pre_}{type_}_{indicator}_3"),
                # With 'question_' prefix
                "q_idea_a": row.get("target_paper_research_idea"),  
                "q_idea_b": row.get(f"{pre_}{type_1}_{indicator}_1"),
                "q_idea_c": row.get(f"{pre_}{type_1}_{indicator}_2"),
                "q_idea_d": row.get(f"{pre_}{type_1}_{indicator}_3")
            }

            papers.append(paper_record)

        return {"papers": papers}

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return {"papers": [], "question": None}
    except Exception as e:
        print(f"Error while loading data: {e}")
        return {"papers": [], "question": None}

if __name__ == "__main__":
    data = load_initial_data()
    papers = data["papers"]
    question = data["question"]
    print("Loaded papers:", len(papers))
    print("Example question:", papers[0]['questions'][1])