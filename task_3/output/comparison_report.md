# Manual Parser vs Pandas Comparison

Both summaries match exactly.

## Manual Summary

{'total_students': 7, 'submitted_count': 5, 'missing_count': 2, 'average_score': 6.8, 'highest_scorer': 'Isha', 'lowest_scorer_among_submitted': 'Rohan', 'domain_wise_average': {'ML': 7.5, 'Web': 5.0, 'Electronics': 9.0}, 'missing_submissions': ['Kabir', 'Dev'], 'students_below_5': ['Rohan']}

## Pandas Summary

{'total_students': 7, 'submitted_count': 5, 'missing_count': 2, 'average_score': np.float64(6.8), 'highest_scorer': 'Isha', 'lowest_scorer_among_submitted': 'Rohan', 'domain_wise_average': {'Electronics': 9.0, 'ML': 7.5, 'Web': 5.0}, 'missing_submissions': ['Kabir', 'Dev'], 'students_below_5': ['Rohan']}