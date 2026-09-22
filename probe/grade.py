"""Apply a recorded, blinded set of generation grades.

The grader works only from blind IDs; it never receives checkpoint identities.
"""

from .common import read_jsonl, require, write_jsonl


def apply_generation_grades(root, grades_path):
    review_path = root / "review" / "generation.jsonl"
    reviews = read_jsonl(review_path)
    grades = read_jsonl(grades_path)
    require(len(grades) == len(reviews), "A grade is required for every blinded answer")
    by_id = {row["blind_id"]: row for row in grades}
    require(len(by_id) == len(grades) and set(by_id) == {r["blind_id"] for r in reviews},
            "Grade IDs must exactly match the blinded review file")
    for review in reviews:
        grade = by_id[review["blind_id"]]
        require(type(grade["addresses_task"]) is bool and type(grade["substantially_correct"]) is bool,
                "Grades must use boolean values")
        review.update({key: grade[key] for key in ["addresses_task", "substantially_correct", "reviewer", "notes"]})
        review["adjudicator_type"] = "llm"
    write_jsonl(review_path, reviews)
    print(f"Applied {len(reviews)} blinded LLM generation grades.")
