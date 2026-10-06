# homeowners - core fields (SPEC_21 4.3)

Core field list of the homeowners SPEC_21 schema, built from all 37 seed golds of 11 carriers (Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `homeowners.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 177 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 122 of the 177 |
| Seeds | 37 seed golds, 11 carriers: Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers |
| Blocks besides the envelope | `locations`, `buildings`, `scheduled_items`, `rating_modifiers` |
| LOB block | none |
| Coverage codes | 46 (`homeowners.coverage_codes.yaml`), of them 8 shared `X_` codes |
| Mandatory fields | `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[].limits[].amount` |

Field names, meanings and shapes are fixed by the common model; this line chooses the blocks, the coverage
codes and the printed labels below. **Printed labels** are the captions the line's seeds print beside or above
the value (read from the seed text layers and OCR transcriptions, checked by hand); they are the schema's
`fideon:aliases` for the field. The common model's own aliases for the field still apply and are not repeated.
**Carriers** and **Seeds** count the seed golds that fill the field. Coverage names are aliases of the
coverage codes, not of `coverages[].coverage_name`, and are listed in the coverage codes file.

## Core fields

| Field | Meaning | Printed labels (this line) | Carriers that print it | Seeds |
|---|---|---|---|---|
| `document.transaction_type` | Kind of transaction the document records. | Transaction Type | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 33 |
| `document.transaction_date` | Date the transaction was processed. | TRANSACTION DATE | Travelers | 1 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | Effective Date of Change; Amended Declarations Page as of | Dryden Mutual, Midstate Mutual Insurance Company, NYCM Insurance, Plymouth, Travelers | 6 |
| `document.transaction_reason` | Reason for the transaction, as printed. | Transaction Reason | Dryden Mutual, NYCM Insurance, Plymouth | 3 |
| `document.endorsement_number` | Number of the endorsement or policy change. | Term-Seq | Dryden Mutual | 1 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `document.issue_date` | Date the document was issued. | Issue Date; Issued On Date; Amended Date | Madison Mutual, Midstate Mutual Insurance Company, Plymouth, Travelers | 23 |
| `document.print_date` | Date the document was printed. | PRINT DATE; Print Date; Printed on; Process Date | Dryden Mutual, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 8 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | Date Mailed | Mercury Insurance Company | 2 |
| `carrier.name` | The writing company. | Company Name; Insurer; Your Insurer | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `carrier.naic_code` | NAIC company code of the writing company. | NAIC | Travelers | 1 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Millennial Specialty Insurance, North Country Insurance Company, Plymouth | 28 |
| `carrier.address.city` | City. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `carrier.address.postal_code` | ZIP code. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 35 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | Chubb, Leatherstocking Cooperative Insurance Company, Madison Mutual, NYCM Insurance | 23 |
| `carrier.fax` | Fax number. | - | Leatherstocking Cooperative Insurance Company | 2 |
| `carrier.web_address` | Website address. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, NYCM Insurance, Plymouth | 27 |
| `carrier.claims_phone` | Phone number for reporting a claim. | Claims Contact; For Claim Service; To Report a Claim; To report a claim please call | Chubb, Mercury Insurance Company, Millennial Specialty Insurance, Plymouth, Travelers | 7 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `producer.producer_code` | Code the carrier assigns to the agency. | AGENCY CODE; PRODUCER SUB-CODE | Dryden Mutual, Madison Mutual, Mercury Insurance Company, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 27 |
| `producer.contact_name` | Contact person at the agency. | - | Dryden Mutual | 2 |
| `producer.address.street` | Street address line. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, Plymouth | 22 |
| `producer.address.city` | City. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `producer.address.postal_code` | ZIP code. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | Agent Phone; Phone Number; For Policy Service | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 35 |
| `producer.fax` | Fax number. | Fax | Leatherstocking Cooperative Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, Travelers | 5 |
| `producer.email` | Email address. | Email Address | Dryden Mutual, NYCM Insurance, North Country Insurance Company | 4 |
| `producer.web_address` | Website of the agency. | - | not printed on the current seeds | 0 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured; Named Insured and Mailing Address; Insured Name; APPLICANT NAME AND ADDRESS | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | Dryden Mutual, Madison Mutual, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 18 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | NYCM Insurance | 1 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | NYCM Insurance | 1 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | NYCM Insurance | 1 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | NYCM Insurance | 1 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 2 |
| `named_insured.mailing_address.city` | City. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | - | Dryden Mutual, NYCM Insurance, Travelers | 4 |
| `named_insured.email` | Email address of the insured. | - | Dryden Mutual, NYCM Insurance, Travelers | 4 |
| `named_insured.date_of_birth` | Date of birth. | DOB; Date of Birth | Dryden Mutual, NYCM Insurance, Travelers | 4 |
| `named_insured.gender` | Gender as printed. | - | NYCM Insurance | 1 |
| `named_insured.marital_status` | Marital status as printed. | - | NYCM Insurance | 1 |
| `named_insured.occupation` | Occupation as printed. | - | NYCM Insurance | 1 |
| `policy.policy_number` | Policy number. | Policy Number; POLICY NUMBER; Policy ID; POLICY #; Policy #; Your Policy Number | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 35 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | POLICY TYPE | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Millennial Specialty Insurance, Plymouth, Travelers | 33 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | Policyholder Since; INCEPTION DATE | Madison Mutual, Travelers | 20 |
| `policy.effective_date` | Date coverage starts. | Policy Effective Date; Effective Date; Policy Term Effective Date; EFFECTIVE DATE; POLICY PERIOD FROM | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `policy.expiration_date` | Date coverage ends. | Policy Expiration Date; EXPIRATION DATE; Expiration Date | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | - | not printed on the current seeds | 0 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | - | not printed on the current seeds | 0 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | not printed on the current seeds | 0 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | Leatherstocking Cooperative Insurance Company, NYCM Insurance | 2 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | Madison Mutual, Mercury Insurance Company, North Country Insurance Company, Plymouth, Travelers | 27 |
| `coverages[].limits[].amount` | Amount of the limit. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | Mercury Insurance Company, Travelers | 3 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, Travelers | 23 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | Deductible; Property Coverage Deductible | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Midstate Mutual Insurance Company, North Country Insurance Company, Travelers | 25 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `coverages[].premium` | Premium for the coverage. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, North Country Insurance Company, Plymouth, Travelers | 35 |
| `coverages[].valuation` | Valuation basis. | Loss Settlement; Loss Settlement Building; Loss Settlement Contents; Rating Basis - Personal Property; Rating Basis - Residence | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, North Country Insurance Company | 26 |
| `coverages[].coinsurance_percent` | Coinsurance percentage that applies to the coverage. | - | not printed on the current seeds | 0 |
| `coverages[].covered_auto_symbols[]` | ISO covered-auto symbols printed against the coverage (1, 2, 7, 8, 9...). Business auto, garage, truckers and package auto parts. | - | not printed on the current seeds | 0 |
| `deductibles[].amount` | Amount of a flat deductible. | Deductible; All Other Perils Deductible; All Other Perils | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 14 |
| `deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | Named Storm Deductible | Millennial Specialty Insurance | 1 |
| `deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | Chubb, Mercury Insurance Company, Millennial Specialty Insurance, Plymouth, Travelers | 6 |
| `interested_parties[].rank` | 1 for first mortgagee, 2 for second... | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, Travelers | 10 |
| `interested_parties[].name` | Name of the interested party. | 1st Mortgagee; First Mortgagee; ADDITIONAL INTEREST(S); Name and Address of Person or Organization; Loc Mortgagee or Secured Party | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 18 |
| `interested_parties[].address.street` | Street address line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 18 |
| `interested_parties[].address.street_2` | Second address line: suite, unit, attention line. | - | Leatherstocking Cooperative Insurance Company, NYCM Insurance | 3 |
| `interested_parties[].address.city` | City. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 18 |
| `interested_parties[].address.state` | State, two-letter USPS code in parsed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 18 |
| `interested_parties[].address.postal_code` | ZIP code. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 18 |
| `interested_parties[].address.county` | County. | - | not printed on the current seeds | 0 |
| `interested_parties[].loan_number` | Loan or account number with the interested party. | Loan Number; LOAN NUMBER | Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, Travelers | 9 |
| `interested_parties[].is_payor` | True when this party pays the premium (escrow billing). | Payor | Mercury Insurance Company, North Country Insurance Company | 3 |
| `premium.total` | Total premium for the policy. | Total Policy Premium; Annual Policy Premium; PREMIUM AT INCEPTION; Premium Amount | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | - | not printed on the current seeds | 0 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Premium adjustment for this change | Dryden Mutual | 1 |
| `premium.items[].description` | Description of the premium line as printed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, North Country Insurance Company, Plymouth | 28 |
| `premium.items[].amount` | Premium amount of the line. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, North Country Insurance Company, Plymouth | 28 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Leatherstocking Cooperative Insurance Company, Madison Mutual | 21 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | Fees | Leatherstocking Cooperative Insurance Company, Madison Mutual | 21 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Chubb, Madison Mutual, Midstate Mutual Insurance Company | 12 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | NJ Property-Liability Guaranty Association Surcharge | Chubb, Madison Mutual, Midstate Mutual Insurance Company | 12 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, North Country Insurance Company, Travelers | 32 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Midstate Mutual Insurance Company, North Country Insurance Company, Travelers | 28 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | Chubb | 1 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | Policy Pay Plan | Dryden Mutual, Plymouth | 7 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | Dryden Mutual | 4 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | not printed on the current seeds | 0 |
| `billing.account_number` | Billing account number the carrier bills under. | BILLING ACCOUNT # | Travelers | 1 |
| `billing.amount_due` | Amount currently due on the bill. | - | Travelers | 1 |
| `billing.due_date` | Date payment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].amount` | Amount of the installment. | - | not printed on the current seeds | 0 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 30 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 27 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 19 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Midstate Mutual Insurance Company, North Country Insurance Company, Plymouth, Travelers | 12 |
| `locations[].location_number` | Location number as printed on the schedule. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Midstate Mutual Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 28 |
| `locations[].address.street` | Street address line. | Insured Property Location | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `locations[].address.city` | City. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | Chubb, Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 37 |
| `locations[].address.postal_code` | ZIP code. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Plymouth, Travelers | 36 |
| `locations[].address.county` | County. | County; COUNTY | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company | 29 |
| `locations[].territory` | Rating territory or zone. | Territory | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company, Travelers | 8 |
| `locations[].protection_class` | Fire protection class of the premises. | Protection Class; Fire Protection; Fire Protection Class | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Travelers | 31 |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | Fire District | Dryden Mutual, Midstate Mutual Insurance Company, NYCM Insurance, Travelers | 7 |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | not printed on the current seeds | 0 |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | Distance to Station; Miles From Fire Dept | Dryden Mutual, Travelers | 5 |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | Distance to Hydrant; Feet From Hydrant | Dryden Mutual, Midstate Mutual Insurance Company, NYCM Insurance, Travelers | 7 |
| `buildings[].building_number` | Building number as printed on the schedule. | - | North Country Insurance Company | 1 |
| `buildings[].construction` | Construction type (frame, masonry, fire resistive...). | Construction; Construction Type | Dryden Mutual, Madison Mutual, Mercury Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Travelers | 31 |
| `buildings[].occupancy` | How the building is occupied or used. | Occupancy; Residence Type Code | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Midstate Mutual Insurance Company, Millennial Specialty Insurance, North Country Insurance Company, Travelers | 10 |
| `buildings[].year_built` | Year the building was built. | Year Built; Year of Construction | Dryden Mutual, Madison Mutual, Mercury Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Travelers | 29 |
| `buildings[].area_sq_ft` | Floor area in square feet. | Square Footage | Mercury Insurance Company, Millennial Specialty Insurance, Travelers | 4 |
| `buildings[].stories` | Number of stories. | # of Stories | Travelers | 1 |
| `buildings[].number_of_units` | Number of dwelling units or families in the building. | # of Families | Dryden Mutual, Midstate Mutual Insurance Company, Millennial Specialty Insurance, NYCM Insurance, North Country Insurance Company, Travelers | 6 |
| `buildings[].roof_type` | Roof type or covering material. | Roof Type; Roof Material Type | Dryden Mutual, Travelers | 3 |
| `buildings[].protective_devices[]` | Protective devices fitted (alarms, sprinklers, extinguishers...). | Protective Devices; Premises Alarm; Fire Alarm | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 3 |
| `buildings[].basement` | Whether the building has a basement. | - | not printed on the current seeds | 0 |
| `buildings[].sprinklered` | Whether the building has an automatic sprinkler system. | Sprinklers | North Country Insurance Company | 1 |
| `buildings[].heating_type` | Type of the primary heating system. | Primary Heating Type | Dryden Mutual | 2 |
| `buildings[].heating_fuel` | Fuel of the primary heating system. | PRIMARY HEAT SOURCE | NYCM Insurance | 1 |
| `buildings[].heating_year_updated` | Year the primary heating system was last updated. | - | not printed on the current seeds | 0 |
| `scheduled_items[].item_number` | Item number as printed on the schedule. | - | NYCM Insurance | 1 |
| `scheduled_items[].category` | Class or category of the scheduled item (jewelry, fine arts, cameras...). | - | NYCM Insurance | 1 |
| `scheduled_items[].description` | Description of the item as printed. | - | NYCM Insurance | 1 |
| `scheduled_items[].serial` | Serial number. | - | not printed on the current seeds | 0 |
| `scheduled_items[].value` | Insured value of the item. | - | NYCM Insurance | 1 |
| `scheduled_items[].valuation` | Valuation basis of the item. | - | not printed on the current seeds | 0 |
| `rating_modifiers[].description` | As printed, e.g. 'Experience Modification', 'Multi-Policy Discount'. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Midstate Mutual Insurance Company, NYCM Insurance | 4 |
| `rating_modifiers[].factor` | Modification factor (e.g. 0.85). | - | not printed on the current seeds | 0 |
| `rating_modifiers[].percent` | Modifier as a percentage. | - | Dryden Mutual, NYCM Insurance | 2 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (9 today: 1st Mortgagee; Agent; Coverage Premium; Form; Interest; Limit; Policy Term Effective Date; Premium; We will pay up to).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| Interest | 4 | Dryden Mutual, Leatherstocking Cooperative Insurance Company, Madison Mutual, NYCM Insurance |
| 1st Mortgagee | 2 | Leatherstocking Cooperative Insurance Company, Madison Mutual |
| Agent | 2 | Dryden Mutual, Plymouth |
| Coverage Premium | 2 | Leatherstocking Cooperative Insurance Company, Madison Mutual |
| Form | 2 | Leatherstocking Cooperative Insurance Company, Travelers |
| Limit | 2 | Dryden Mutual, Madison Mutual |
| Policy Term Effective Date | 2 | Leatherstocking Cooperative Insurance Company, Madison Mutual |
| Premium | 2 | Madison Mutual, Midstate Mutual Insurance Company |
| We will pay up to | 2 | Chubb, Millennial Specialty Insurance |
| # of Bathrooms | 1 | Travelers |
| # of Employees | 1 | Travelers |
| (2) \| or upon "motor vehicle". shi | 1 | Millennial Specialty Insurance |
| 1. We insure, for up to | 1 | Millennial Specialty Insurance |
| 1. We will pay up to | 1 | Millennial Specialty Insurance |
| 2) For which no insured receives more than | 1 | Millennial Specialty Insurance |
| 2) Less than | 1 | Millennial Specialty Insurance |
| 2. We do not cover | 1 | Millennial Specialty Insurance |
| a. On the residence premises for the home day care "business" to | 1 | Millennial Specialty Insurance |
| a. We will pay up to | 1 | Millennial Specialty Insurance |
| Account Number | 1 | Plymouth |
| Acreage | 1 | Madison Mutual |
| Additional Interest | 1 | Leatherstocking Cooperative Insurance Company |
| ADDITIONAL INTEREST(S) | 1 | Leatherstocking Cooperative Insurance Company |
| Age of Home | 1 | Dryden Mutual |
| Age of Roof | 1 | Travelers |
| ALTERNATE HEAT SOURCE | 1 | NYCM Insurance |
| Amended Date | 1 | Leatherstocking Cooperative Insurance Company |
| Amount of Increase | 1 | Madison Mutual |
| Amount of itemized coverage | 1 | Chubb |
| AMOUNT PAID | 1 | Travelers |
| Annual Change in Premium | 1 | Madison Mutual |
| annual premium savings | 1 | Chubb |
| Any Water Exposures? | 1 | Dryden Mutual |
| Applicable Coverage(s) | 1 | Madison Mutual |
| Applicant | 1 | Dryden Mutual |
| Applies to | 1 | Dryden Mutual |
| Assigned To | 1 | North Country Insurance Company |
| Authorized representative | 1 | Chubb |
| Authorized Representative | 1 | Plymouth |
| Auto Increase in Insurance | 1 | Dryden Mutual |
| Base deductible Waived for losses greater than: | 1 | Chubb |
| Basement | 1 | Travelers |
| Basement Construction | 1 | Madison Mutual |
| Billing Information | 1 | Dryden Mutual |
| Broker Fee | 1 | Millennial Specialty Insurance |
| but not loss of other income) up to | 1 | Millennial Specialty Insurance |
| but not more than | 1 | Chubb |
| by calling | 1 | Chubb |
| Call | 1 | Millennial Specialty Insurance |
| cards, or gift certificates | 1 | Chubb |
| Cause of Loss Form | 1 | Dryden Mutual |
| Causes of Loss Form | 1 | Madison Mutual |
| Change In Annual Premium | 1 | Dryden Mutual |
| CODED SCORE | 1 | NYCM Insurance |
| Construction Year | 1 | Leatherstocking Cooperative Insurance Company |
| Contents | 1 | Chubb |
| CONTRACT # | 1 | Travelers |
| Cost of Improvements | 1 | Dryden Mutual |
| Coverage Level | 1 | Travelers |
| Coverage M-Medical Payments to Others is increased by | 1 | Midstate Mutual Insurance Company |
| covered loss of | 1 | Millennial Specialty Insurance |
| Date | 1 | Dryden Mutual |
| Date Paid | 1 | Travelers |
| Deductible Type | 1 | Madison Mutual |
| Delivery Preference | 1 | Dryden Mutual |
| Deluxe Contents Coverage footer print time | 1 | Chubb |
| Deluxe House Coverage / Deluxe Contents Coverage footer edition | 1 | Chubb |
| Deluxe House Coverage footer print time | 1 | Chubb |
| Describe | 1 | Dryden Mutual |
| Describe the breed of each dog on premises | 1 | Dryden Mutual |
| described in A. above is more than | 1 | Millennial Specialty Insurance |
| Description | 1 | North Country Insurance Company |
| Description of Structure | 1 | Madison Mutual |
| Device | 1 | Leatherstocking Cooperative Insurance Company |
| DOB | 1 | Dryden Mutual |
| Document History | 1 | Dryden Mutual |
| Document Reference | 1 | Dryden Mutual |
| Document Region | 1 | Dryden Mutual |
| Document Title | 1 | Dryden Mutual |
| does not exceed | 1 | Chubb |
| Dog # 1 | 1 | Dryden Mutual |
| DOUBLEWIDE | 1 | NYCM Insurance |
| Dwelling | 1 | Chubb |
| Dwelling Use | 1 | Travelers |
| Electrical Service | 1 | Dryden Mutual |
| EMPLOYMENT STATUS | 1 | NYCM Insurance |
| exceed | 1 | Millennial Specialty Insurance |
| Excess Line Association of New York | 1 | Millennial Specialty Insurance |
| Expiration Date | 1 | North Country Insurance Company |
| Fax: | 1 | Chubb |
| File # | 1 | Dryden Mutual |
| financial institutions or similar credit grantors We will pay up to | 1 | Millennial Specialty Insurance |
| Finished Basement | 1 | Travelers |
| First Mortgagee | 1 | Dryden Mutual |
| for every | 1 | Chubb |
| for the amount of | 1 | Plymouth |
| FORM DETAILS | 1 | NYCM Insurance |
| FOUNDATION TYPE | 1 | NYCM Insurance |
| Foundation Type | 1 | Travelers |
| FUEL TANK LOCATION | 1 | NYCM Insurance |
| Fuel Type | 1 | Dryden Mutual |
| Garage - Number of Cars | 1 | Travelers |
| Garage Number of Cars | 1 | Travelers |
| Garage Type | 1 | Travelers |
| gross revenues in excess of | 1 | Chubb |
| Hip Roof | 1 | Millennial Specialty Insurance |
| Homeowners | 1 | Travelers |
| How many years has the applicant owned this risk? | 1 | Dryden Mutual |
| If no, please explain | 1 | Dryden Mutual |
| If the covered loss to the dwelling is | 1 | Millennial Specialty Insurance |
| if the limit of liability shown in the Coverage Summary is | 1 | Chubb |
| Incidental Business | 1 | Midstate Mutual Insurance Company |
| incorporated in | 1 | Chubb |
| increase the Coverage C limit of liability. | 1 | Millennial Specialty Insurance |
| Inland Marine Premium | 1 | Dryden Mutual |
| Inspection Fee | 1 | Millennial Specialty Insurance |
| Insured | 1 | Mercury Insurance Company |
| Insured Name | 1 | Dryden Mutual |
| Interest Type | 1 | North Country Insurance Company |
| Item Premium | 1 | Dryden Mutual |
| LexisNexis Consumer Center | 1 | Chubb |
| Liability Coverage | 1 | Leatherstocking Cooperative Insurance Company |
| Limited Water Back-up And Sump Discharge Or Overflow Coverage Limit Of | 1 | Millennial Specialty Insurance |
| LOC TYPE | 1 | NYCM Insurance |
| LOCATION 1 | 1 | Travelers |
| Loss Settlement | 1 | Leatherstocking Cooperative Insurance Company |
| Loss Settlement Building | 1 | Dryden Mutual |
| Mail To | 1 | Madison Mutual |
| Marital Status | 1 | Dryden Mutual |
| Market Value | 1 | Dryden Mutual |
| ML-243 | 1 | Dryden Mutual |
| ML-243 Auto Increase in Insurance | 1 | Dryden Mutual |
| Modifies Coverage(s) at Renewal | 1 | Leatherstocking Cooperative Insurance Company |
| MONTHS OCCUPIED ANNUALLY | 1 | NYCM Insurance |
| Mortgagee | 1 | Leatherstocking Cooperative Insurance Company |
| Mortgagee Company Name | 1 | Plymouth |
| Mortgagee Information | 1 | Dryden Mutual |
| Name of Payer | 1 | Plymouth |
| NAME PRINTED ON DEC PAGES | 1 | Travelers |
| NET CHANGE AMOUNT | 1 | Travelers |
| Number of Apartments | 1 | Travelers |
| Number of Weeks Rented | 1 | Madison Mutual |
| Occupancy Type | 1 | Madison Mutual |
| Occupation | 1 | Dryden Mutual |
| of the total policy premium or | 1 | Millennial Specialty Insurance |
| or by calling | 1 | Chubb |
| or call | 1 | Plymouth |
| ORDERED DATE | 1 | NYCM Insurance |
| Other Coverage(s) As Specified | 1 | Madison Mutual |
| Our payment is subject to a | 1 | Chubb |
| Overall condition of the risk | 1 | Dryden Mutual |
| Paid By | 1 | Mercury Insurance Company |
| Participants | 1 | Dryden Mutual |
| payable that exceeds | 1 | Millennial Specialty Insurance |
| Payment Confirmation | 1 | Plymouth |
| Payment Date/Time | 1 | Plymouth |
| Payment Status | 1 | Plymouth |
| Payment Type | 1 | Plymouth |
| Period | 1 | Travelers |
| Phone | 1 | Dryden Mutual |
| Please explain "Other" | 1 | Dryden Mutual |
| POL COMM | 1 | Travelers |
| POLICY CHANGE | 1 | Leatherstocking Cooperative Insurance Company |
| Policy Form | 1 | Madison Mutual |
| Policy Information Notice footer print time | 1 | Chubb |
| Policy Period | 1 | Dryden Mutual |
| Policy Terms / Policy Information Notice footer edition | 1 | Chubb |
| Policy Terms footer print time | 1 | Chubb |
| Policy Tier | 1 | Madison Mutual |
| Pool | 1 | Travelers |
| PRA 517 | 1 | Plymouth |
| Premises Alarm | 1 | Dryden Mutual |
| Previous Insurance Carrier | 1 | Dryden Mutual |
| Previous Policy Expiration Date | 1 | Dryden Mutual |
| PRH EOI | 1 | Plymouth |
| Primary Heat Source | 1 | Travelers |
| PRIMARY POLICY NUMBER | 1 | NYCM Insurance |
| Principal Unit-at-Risk | 1 | Travelers |
| Prior Annual Premium | 1 | Dryden Mutual |
| Pro-Rata Change in Premium | 1 | Madison Mutual |
| Protection Class | 1 | Leatherstocking Cooperative Insurance Company |
| Protective Device | 1 | Millennial Specialty Insurance |
| Protective Devices | 1 | Dryden Mutual |
| Purchase Price | 1 | Dryden Mutual |
| Questions? Call | 1 | Chubb |
| QUOTE # | 1 | NYCM Insurance |
| Quoted Pro-rated Premium Amount | 1 | NYCM Insurance |
| Rating Zone | 1 | Dryden Mutual |
| Reason for Premium Adjustment | 1 | Madison Mutual |
| receives more than | 1 | Millennial Specialty Insurance |
| Recreational Vehicles (Other Than Watercraft) Offered with Rental | 1 | Madison Mutual |
| Reference Number | 1 | Chubb |
| Remarks | 1 | Travelers |
| responsible for the balance of the covered loss, or | 1 | Millennial Specialty Insurance |
| Revised Annual Premium | 1 | Dryden Mutual |
| Risk Description | 1 | Dryden Mutual |
| Risk Type | 1 | Dryden Mutual |
| Roof to Wall Attachment | 1 | Millennial Specialty Insurance |
| Roof Type | 1 | Travelers |
| Savings Reflected in Your Total Premium | 1 | Travelers |
| SECONDARY INDICATOR | 1 | NYCM Insurance |
| Sender Email | 1 | Dryden Mutual |
| Sender Name | 1 | Dryden Mutual |
| Siding Type | 1 | Travelers |
| Signatures footer print time | 1 | Chubb |
| Slab | 1 | Travelers |
| Smoke Alarm | 1 | Travelers |
| Solid Fuel Burning Device(s) | 1 | Madison Mutual |
| SSN | 1 | NYCM Insurance |
| Stamping Fee | 1 | Millennial Specialty Insurance |
| Summary of Changes | 1 | Plymouth |
| Surplus Lines Producer License Number | 1 | Millennial Specialty Insurance |
| Surplus Lines Producer Name and Address | 1 | Millennial Specialty Insurance |
| Surplus Lines Tax | 1 | Millennial Specialty Insurance |
| Swimming Pool | 1 | Madison Mutual |
| SWIMMING POOL | 1 | NYCM Insurance |
| Telephone: | 1 | Chubb |
| TENANTS HOMEOWNERS UNDERWRITING | 1 | Travelers |
| TERM AMOUNT | 1 | Travelers |
| Term-Seq | 1 | Dryden Mutual |
| Territorial Zone | 1 | Leatherstocking Cooperative Insurance Company |
| TERRITORY | 1 | NYCM Insurance |
| that exceeds | 1 | Millennial Specialty Insurance |
| The | 1 | Millennial Specialty Insurance |
| The NFIP may be contacted by phone at | 1 | Millennial Specialty Insurance |
| the time of a covered loss or | 1 | Chubb |
| Tier | 1 | NYCM Insurance |
| Total Annual Policy Premium | 1 | Dryden Mutual |
| Total Annual Premium for Location # 1 | 1 | Dryden Mutual |
| Total Annual Premium This Location | 1 | Dryden Mutual |
| Total Annual Premium This Location: | 1 | Dryden Mutual |
| Total Document Pages | 1 | Dryden Mutual |
| TOTAL DUE | 1 | Millennial Specialty Insurance |
| TOTAL IM SCHEDULE | 1 | NYCM Insurance |
| Total Limit | 1 | Madison Mutual |
| Total Number of Risks | 1 | Dryden Mutual |
| total of | 1 | Chubb |
| Total Quoted Premium | 1 | Dryden Mutual |
| Total return premium | 1 | Plymouth |
| Trampoline on Premises | 1 | Madison Mutual |
| Type of Home | 1 | Madison Mutual |
| up residence to | 1 | Millennial Specialty Insurance |
| up to a maximum of | 1 | Chubb |
| up to a maximum payment of | 1 | Millennial Specialty Insurance |
| Usage | 1 | North Country Insurance Company |
| Valuable Articles Coverage footer edition | 1 | Chubb |
| Valuable Articles Coverage footer print time | 1 | Chubb |
| Vandalism | 1 | Leatherstocking Cooperative Insurance Company |
| Visibility | 1 | Travelers |
| Water Back Up And Sump Discharge Or Overflow Coverage Deductible | 1 | Madison Mutual |
| Water Leak Detection | 1 | Mercury Insurance Company |
| Watercraft Offered With Rental | 1 | Madison Mutual |
| What type of system? | 1 | Dryden Mutual |
| Who? | 1 | Dryden Mutual |
| Year Roof Updated | 1 | Dryden Mutual |
| you can call this telephone number for further assistance: | 1 | Chubb |
| you incurred, up to | 1 | Chubb |
| Your insurance cost could have been | 1 | Chubb |
| Your premium was reduced to | 1 | Chubb |
