import pytest
import csv
from pathlib import Path
from benchmarks.e2e_facts import (
    parse_mr,
    classify_slot,
    score_example,
    score_dataset,
    export_human_validation,
    compute_human_agreement,
)


def test_parse_mr_standard():
    mr = "name[The Eagle], eatType[coffee shop], food[French], priceRange[moderate]"
    slots = parse_mr(mr)
    assert slots["name"] == "The Eagle"
    assert slots["eatType"] == "coffee shop"
    assert slots["food"] == "French"
    assert slots["priceRange"] == "moderate"
    assert len(slots) == 4


def test_parse_mr_empty_or_malformed():
    assert parse_mr("") == {}
    assert parse_mr("Just plain text with no brackets") == {}


def test_classify_name_and_near():
    assert classify_slot("name", "The Eagle", "Welcome to The Eagle in town.") == "expressed_correct"
    assert classify_slot("name", "The Eagle", "Welcome to the eagle in town.") == "expressed_correct"
    assert classify_slot("name", "The Eagle", "Welcome to The Mill in town.") == "omitted"

    assert classify_slot("near", "Burger King", "Located near Burger King.") == "expressed_correct"
    assert classify_slot("near", "Burger King", "Located near McDonalds.") == "omitted"


def test_classify_eattype_aliases_and_contradiction():
    # Correct alias
    assert classify_slot("eatType", "coffee shop", "A cozy cafe near the river.") == "expressed_correct"
    assert classify_slot("eatType", "coffee shop", "A great coffee shop.") == "expressed_correct"
    # Contradiction
    assert classify_slot("eatType", "coffee shop", "A lively pub in the city.") == "contradicted"
    # Omitted
    assert classify_slot("eatType", "coffee shop", "Serving delicious breakfast.") == "omitted"


def test_classify_food_aliases_and_contradiction():
    assert classify_slot("food", "English", "Serving traditional British cuisine.") == "expressed_correct"
    assert classify_slot("food", "French", "Specializing in French dishes.") == "expressed_correct"
    assert classify_slot("food", "French", "Specializing in Italian pasta.") == "contradicted"
    assert classify_slot("food", "Japanese", "Great food and service.") == "omitted"


def test_classify_area_aliases_and_contradiction():
    assert classify_slot("area", "riverside", "Located by the river.") == "expressed_correct"
    assert classify_slot("area", "riverside", "Located on the riverside.") == "expressed_correct"
    assert classify_slot("area", "riverside", "Located in the city centre.") == "contradicted"
    assert classify_slot("area", "riverside", "Located near the park.") == "omitted"


def test_classify_price_range_buckets():
    assert classify_slot("priceRange", "cheap", "An inexpensive venue with good food.") == "expressed_correct"
    assert classify_slot("priceRange", "cheap", "Meals cost under £20.") == "expressed_correct"
    assert classify_slot("priceRange", "cheap", "An expensive restaurant.") == "contradicted"
    assert classify_slot("priceRange", "expensive", "High priced fine dining.") == "expressed_correct"
    assert classify_slot("priceRange", "moderate", "Casual dining spot.") == "omitted"


def test_classify_customer_rating_buckets():
    assert classify_slot("customer rating", "high", "A top rated venue with 5 out of 5 stars.") == "expressed_correct"
    assert classify_slot("customer rating", "high", "A poor restaurant with 1 out of 5.") == "contradicted"
    assert classify_slot("customer rating", "low", "1 out of 5 customer rating.") == "expressed_correct"
    assert classify_slot("customer rating", "average", "Clean and pleasant atmosphere.") == "omitted"


def test_classify_family_friendly_yes():
    # Positive -> correct
    assert classify_slot("familyFriendly", "yes", "The place is family-friendly and children welcome.") == "expressed_correct"
    # Negative -> contradicted
    assert classify_slot("familyFriendly", "yes", "This is an adults only venue, no children.") == "contradicted"
    # Neither -> omitted
    assert classify_slot("familyFriendly", "yes", "The Eagle is a coffee shop.") == "omitted"


def test_classify_family_friendly_no():
    # Negative -> correct
    assert classify_slot("familyFriendly", "no", "Adults only, not suitable for children.") == "expressed_correct"
    # Positive -> contradicted (inversion)
    assert classify_slot("familyFriendly", "no", "A very family friendly establishment with a play area.") == "contradicted"
    # Neither -> omitted (REGRESSION TEST: must not return expressed_correct)
    assert classify_slot("familyFriendly", "no", "The Eagle is a coffee shop near the river.") == "omitted"


def test_score_example_fully_correct():
    mr = "name[The Eagle], eatType[coffee shop], food[French], familyFriendly[yes]"
    text = "The Eagle is a French coffee shop that is family-friendly."
    res = score_example(mr, text)

    assert res["slots_count"] == 4
    assert res["correct_count"] == 4
    assert res["contradiction_count"] == 0
    assert res["required_detail_recall"] == 1.0
    assert res["fully_correct"] is True
    assert res["contradiction_any"] is False
    assert res["family_friendly_inversion"] is False


def test_score_example_with_omission_and_contradiction():
    mr = "name[The Eagle], eatType[coffee shop], food[French], area[riverside], familyFriendly[no]"
    # eatType contradicted (pub), food omitted, familyFriendly inverted (positive mentioned)
    text = "The Eagle is a riverside pub that is family-friendly."
    res = score_example(mr, text)

    assert res["slots_count"] == 5
    assert res["slot_status"]["name"] == "expressed_correct"
    assert res["slot_status"]["eatType"] == "contradicted"
    assert res["slot_status"]["food"] == "omitted"
    assert res["slot_status"]["area"] == "expressed_correct"
    assert res["slot_status"]["familyFriendly"] == "contradicted"
    assert res["contradiction_count"] == 2
    assert res["contradiction_any"] is True
    assert res["family_friendly_inversion"] is True
    assert res["fully_correct"] is False
    assert res["required_detail_recall"] == 0.4  # 2/5


def test_score_dataset_metrics():
    records = [
        {
            "mr": "name[The Mill], eatType[pub], food[English], familyFriendly[yes]",
            "generated": "The Mill is a British pub welcoming children.",
        },
        {
            "mr": "name[Cafe Sicilia], eatType[restaurant], food[Italian], familyFriendly[no]",
            "generated": "Cafe Sicilia is an Italian restaurant that is adults only.",
        },
        {
            "mr": "name[Golden Wok], eatType[restaurant], food[Chinese], area[city centre]",
            "generated": "Golden Wok is a French cafe in town.",
        },
    ]

    summary = score_dataset(records)
    assert summary["count"] == 3
    assert 0.0 <= summary["required_detail_recall_mean"] <= 1.0
    assert len(summary["required_detail_recall_ci95"]) == 2
    assert summary["required_detail_recall_ci95"][0] <= summary["required_detail_recall_ci95"][1]
    assert "contradiction_rate" in summary
    assert "family_friendly_inversion_rate" in summary
    assert "per_slot_contradiction_rates" in summary


def test_human_validation_export_and_agreement(tmp_path: Path):
    records = [
        {"mr": "name[The Eagle], eatType[pub]", "generated": "The Eagle is a pub."},
        {"mr": "name[The Mill], eatType[pub]", "generated": "The Mill is a cafe."},
    ]
    csv_file = tmp_path / "human_validation.csv"
    export_human_validation(records, csv_file, count=2)
    assert csv_file.exists()

    # Read and fill simulated human labels with 100% agreement
    rows = []
    with csv_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r["human_contradiction_any"] = r["scorer_contradiction_any"]
            rows.append(r)

    with csv_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    agreement = compute_human_agreement(str(csv_file))
    assert agreement["annotated_samples"] == 2
    assert agreement["cohen_kappa"] == 1.0
    assert agreement["meets_threshold_0_8"] is True


def test_classify_eattype_pub_and_restaurant():
    assert classify_slot("eatType", "pub", "A lively pub with local ales.") == "expressed_correct"
    assert classify_slot("eatType", "restaurant", "A family restaurant serving dinner.") == "expressed_correct"
    assert classify_slot("eatType", "pub", "A quiet cafe serving tea.") == "contradicted"


def test_classify_food_chinese_and_indian():
    assert classify_slot("food", "Chinese", "Authentic Chinese dumplings and noodles.") == "expressed_correct"
    assert classify_slot("food", "Indian", "Spicy Indian curries and naan.") == "expressed_correct"
    assert classify_slot("food", "Chinese", "Traditional English fish and chips.") == "contradicted"


def test_classify_area_city_center_downtown():
    assert classify_slot("area", "city centre", "Conveniently located downtown.") == "expressed_correct"
    assert classify_slot("area", "city centre", "In the center of the city.") == "expressed_correct"
    assert classify_slot("area", "city centre", "Situated in town near shops.") == "expressed_correct"


def test_classify_price_range_moderate_20_25():
    assert classify_slot("priceRange", "moderate", "Dishes priced £20-25 per person.") == "expressed_correct"
    assert classify_slot("priceRange", "moderate", "Mid-range prices for casual dining.") == "expressed_correct"
    assert classify_slot("priceRange", "expensive", "Average price around £15.") == "contradicted"


def test_classify_customer_rating_3_out_of_5():
    assert classify_slot("customer rating", "average", "Awarded 3 out of 5 stars.") == "expressed_correct"
    assert classify_slot("customer rating", "average", "A moderate rating from diners.") == "expressed_correct"
    assert classify_slot("customer rating", "low", "Received 5/5 stars from guests.") == "contradicted"


def test_classify_family_friendly_is_not_kid_friendly():
    assert classify_slot("familyFriendly", "no", "The bar isn't kid friendly.") == "expressed_correct"
    assert classify_slot("familyFriendly", "no", "It is not child friendly.") == "expressed_correct"
    assert classify_slot("familyFriendly", "yes", "The venue is non-family orientated.") == "contradicted"


def test_classify_custom_fallback_slot():
    assert classify_slot("customSlot", "Outdoor Seating", "Enjoy our outdoor seating area.") == "expressed_correct"
    assert classify_slot("customSlot", "Outdoor Seating", "Indoor dining only.") == "omitted"


def test_score_example_empty_inputs():
    res = score_example("", "")
    assert res["slots_count"] == 0
    assert res["required_detail_recall"] == 0.0
    assert res["fully_correct"] is False

    res2 = score_example("name[The Eagle]", "")
    assert res2["slots_count"] == 1
    assert res2["correct_count"] == 0
    assert res2["required_detail_recall"] == 0.0


def test_score_example_multiple_contradictions():
    mr = "name[The Mill], eatType[pub], food[Indian], priceRange[cheap], familyFriendly[no]"
    # Contradict eatType (cafe), food (Italian), priceRange (expensive), familyFriendly (children welcome)
    text = "The Mill is an expensive Italian cafe where children are welcome."
    res = score_example(mr, text)

    assert res["slots_count"] == 5
    assert res["correct_count"] == 1  # name only
    assert res["contradiction_count"] == 4
    assert res["family_friendly_inversion"] is True
    assert res["contradiction_any"] is True
    assert res["required_detail_recall"] == 0.2


def test_score_dataset_empty_records():
    res = score_dataset([])
    assert res == {"count": 0}


def test_compute_human_agreement_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        compute_human_agreement("non_existent_validation_file_xyz.csv")


def test_compute_human_agreement_empty_file(tmp_path: Path):
    empty_csv = tmp_path / "empty_validation.csv"
    with empty_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "human_contradiction_any", "scorer_contradiction_any"])
        writer.writeheader()

    res = compute_human_agreement(str(empty_csv))
    assert res["annotated_samples"] == 0
    assert res["cohen_kappa"] == 0.0
