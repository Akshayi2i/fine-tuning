"""Pages the router must not skip in a long policy (serving.page_router)."""

from __future__ import annotations

from serving.page_router import SELECTION_THRESHOLD, plan_pages, score_page

#: Chubb's coverage summary continued, and a Travelers limits page: both scored 0.
CONTINUED = ("Coverage Summary Page 2 Effective date 6/4/21 Policy no. 15162088-01 Homes and Contents "
             "(Continued) Water backup deductible In lieu of the base deductible")
LIMITS = ("PL-50355 PA (05-17) Page D-2 Liability Coverage Section Limit Coverage E - Personal Liability "
          "$500,000 Coverage F - Medical Payments to Others $1,000 Peril Deductible All Perils $1,000")


def test_a_coverage_summary_continued_and_a_limits_page_are_read():
    assert score_page(CONTINUED, 21).score >= SELECTION_THRESHOLD
    assert score_page(LIMITS, 2).score >= SELECTION_THRESHOLD
    texts = {1: "Policy Declarations Named Insured", 2: LIMITS, 3: "Fraud notice", 4: "Privacy notice",
             5: "Your online account", 6: CONTINUED, 7: "Signatures"}
    assert plan_pages(texts).pages == [1, 2, 6]


def test_a_notice_page_is_still_skipped():
    notice = "IMPORTANT NOTICE - DRIVING WHILE IMPAIRED - NEW YORK. Overall 33% Nighttime 59% Weekends 45%"
    assert score_page(notice, 23).score < SELECTION_THRESHOLD
