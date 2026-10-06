# personal_umbrella - core fields (SPEC_21 4.3)

Core field list of the personal_umbrella SPEC_21 schema, built from all 4 seed golds of 2 carriers (NYCM Insurance, Plymouth) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `personal_umbrella.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 149 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 60 of the 149 |
| Seeds | 4 seed golds, 2 carriers: NYCM Insurance, Plymouth |
| Blocks besides the envelope | `underlying_insurance` |
| LOB block | none |
| Coverage codes | 3 (`personal_umbrella.coverage_codes.yaml`), of them 0 shared `X_` codes |
| Mandatory fields | `carrier.name`, `named_insured.primary_name`, `policy.policy_number`, `policy.effective_date`, `policy.expiration_date`, `coverages[coverage_code=PU_LIABILITY].limits[limit_type=per_occurrence].amount` |

Field names, meanings and shapes are fixed by the common model; this line chooses the blocks, the coverage
codes and the printed labels below. **Printed labels** are the captions the line's seeds print beside or above
the value (read from the seed text layers and OCR transcriptions, checked by hand); they are the schema's
`fideon:aliases` for the field. The common model's own aliases for the field still apply and are not repeated.
**Carriers** and **Seeds** count the seed golds that fill the field. Coverage names are aliases of the
coverage codes, not of `coverages[].coverage_name`, and are listed in the coverage codes file.

## Core fields

| Field | Meaning | Printed labels (this line) | Carriers that print it | Seeds |
|---|---|---|---|---|
| `document.transaction_type` | Kind of transaction the document records. | Transaction Type | NYCM Insurance, Plymouth | 4 |
| `document.transaction_date` | Date the transaction was processed. | - | not printed on the current seeds | 0 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | Transaction Effective | NYCM Insurance, Plymouth | 3 |
| `document.transaction_reason` | Reason for the transaction, as printed. | Transaction Reason; Transaction Reason Description | NYCM Insurance, Plymouth | 2 |
| `document.endorsement_number` | Number of the endorsement or policy change. | - | not printed on the current seeds | 0 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | NYCM Insurance, Plymouth | 4 |
| `document.issue_date` | Date the document was issued. | Issue Date | Plymouth | 2 |
| `document.print_date` | Date the document was printed. | - | NYCM Insurance | 2 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | not printed on the current seeds | 0 |
| `carrier.name` | The writing company. | Insurance Provided By; Policy Issued by | NYCM Insurance, Plymouth | 4 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | NYCM Insurance, Plymouth | 4 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | Plymouth | 2 |
| `carrier.address.city` | City. | - | NYCM Insurance, Plymouth | 4 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | NYCM Insurance, Plymouth | 4 |
| `carrier.address.postal_code` | ZIP code. | - | NYCM Insurance, Plymouth | 4 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | NYCM Insurance | 2 |
| `carrier.fax` | Fax number. | - | not printed on the current seeds | 0 |
| `carrier.web_address` | Website address. | - | NYCM Insurance, Plymouth | 4 |
| `carrier.claims_phone` | Phone number for reporting a claim. | To Report a Claim | Plymouth | 2 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | - | NYCM Insurance, Plymouth | 4 |
| `producer.producer_code` | Code the carrier assigns to the agency. | - | NYCM Insurance, Plymouth | 4 |
| `producer.contact_name` | Contact person at the agency. | - | not printed on the current seeds | 0 |
| `producer.address.street` | Street address line. | - | NYCM Insurance, Plymouth | 4 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | NYCM Insurance | 1 |
| `producer.address.city` | City. | - | NYCM Insurance, Plymouth | 4 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | NYCM Insurance, Plymouth | 4 |
| `producer.address.postal_code` | ZIP code. | - | NYCM Insurance, Plymouth | 4 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | - | NYCM Insurance, Plymouth | 4 |
| `producer.fax` | Fax number. | - | NYCM Insurance | 2 |
| `producer.email` | Email address. | - | NYCM Insurance | 2 |
| `producer.web_address` | Website of the agency. | - | NYCM Insurance | 2 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured; Named Insured and Address | NYCM Insurance, Plymouth | 4 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | NYCM Insurance, Plymouth | 3 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | NYCM Insurance | 1 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | NYCM Insurance | 2 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | NYCM Insurance | 2 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | NYCM Insurance, Plymouth | 4 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.city` | City. | - | NYCM Insurance, Plymouth | 4 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | NYCM Insurance, Plymouth | 4 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | NYCM Insurance, Plymouth | 4 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | - | not printed on the current seeds | 0 |
| `named_insured.email` | Email address of the insured. | - | not printed on the current seeds | 0 |
| `named_insured.date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.gender` | Gender as printed. | - | NYCM Insurance | 2 |
| `named_insured.marital_status` | Marital status as printed. | - | NYCM Insurance | 2 |
| `named_insured.occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `policy.policy_number` | Policy number. | Policy Number | NYCM Insurance, Plymouth | 4 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | - | NYCM Insurance | 2 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | Inception Date; Protected Since Date | NYCM Insurance | 2 |
| `policy.effective_date` | Date coverage starts. | Effective Date; Policy Effective Date | NYCM Insurance, Plymouth | 4 |
| `policy.expiration_date` | Date coverage ends. | Expiration Date | NYCM Insurance, Plymouth | 4 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | Term Length | NYCM Insurance | 2 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | - | not printed on the current seeds | 0 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | NYCM Insurance | 2 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | NYCM Insurance, Plymouth | 4 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | NYCM Insurance, Plymouth | 4 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].amount` | Amount of the limit. | - | NYCM Insurance, Plymouth | 4 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | Deductible | NYCM Insurance, Plymouth | 4 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `coverages[].premium` | Premium for the coverage. | - | NYCM Insurance, Plymouth | 4 |
| `coverages[].valuation` | Valuation basis. | - | not printed on the current seeds | 0 |
| `coverages[].coinsurance_percent` | Coinsurance percentage that applies to the coverage. | - | not printed on the current seeds | 0 |
| `coverages[].covered_auto_symbols[]` | ISO covered-auto symbols printed against the coverage (1, 2, 7, 8, 9...). Business auto, garage, truckers and package auto parts. | - | not printed on the current seeds | 0 |
| `deductibles[].amount` | Amount of a flat deductible. | - | not printed on the current seeds | 0 |
| `deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `interested_parties[].rank` | 1 for first mortgagee, 2 for second... | - | not printed on the current seeds | 0 |
| `interested_parties[].name` | Name of the interested party. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.city` | City. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.county` | County. | - | not printed on the current seeds | 0 |
| `interested_parties[].loan_number` | Loan or account number with the interested party. | - | not printed on the current seeds | 0 |
| `interested_parties[].is_payor` | True when this party pays the premium (escrow billing). | - | not printed on the current seeds | 0 |
| `premium.total` | Total premium for the policy. | TOTAL PREMIUM; Total Policy Premium | NYCM Insurance, Plymouth | 4 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | - | not printed on the current seeds | 0 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Total return premium | Plymouth | 1 |
| `premium.items[].description` | Description of the premium line as printed. | - | not printed on the current seeds | 0 |
| `premium.items[].amount` | Premium amount of the line. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | - | Plymouth | 2 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | not printed on the current seeds | 0 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | not printed on the current seeds | 0 |
| `billing.account_number` | Billing account number the carrier bills under. | - | not printed on the current seeds | 0 |
| `billing.amount_due` | Amount currently due on the bill. | - | not printed on the current seeds | 0 |
| `billing.due_date` | Date payment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].amount` | Amount of the installment. | - | not printed on the current seeds | 0 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | NYCM Insurance, Plymouth | 4 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | NYCM Insurance, Plymouth | 4 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | NYCM Insurance, Plymouth | 4 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | NYCM Insurance, Plymouth | 4 |
| `underlying_insurance[].carrier_name` | Insurer of the underlying policy. | - | NYCM Insurance, Plymouth | 4 |
| `underlying_insurance[].policy_number` | Policy number of the underlying policy. | - | NYCM Insurance, Plymouth | 4 |
| `underlying_insurance[].coverage_name` | Underlying coverage as printed. | - | NYCM Insurance, Plymouth | 4 |
| `underlying_insurance[].limits[].amount` | Amount of the limit. | - | NYCM Insurance | 2 |
| `underlying_insurance[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `underlying_insurance[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | not printed on the current seeds | 0 |
| `underlying_insurance[].effective_date` | Date the underlying policy starts. | - | not printed on the current seeds | 0 |
| `underlying_insurance[].expiration_date` | Date the underlying policy ends. | - | not printed on the current seeds | 0 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (0 today).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| ADDRESS | 1 | NYCM Insurance |
| Auto Liability and Recreational Vehicle Liability | 1 | Plymouth |
| Automobile Liability | 1 | NYCM Insurance |
| BUSINESS | 1 | NYCM Insurance |
| Business Pursuits/Incidental Occupancies* | 1 | Plymouth |
| City | 1 | NYCM Insurance |
| County Code | 1 | NYCM Insurance |
| County Name | 1 | NYCM Insurance |
| Direct Mail | 1 | NYCM Insurance |
| Effective Date | 1 | NYCM Insurance |
| If you do not cancel your form of payment for your insurance policy by | 1 | Plymouth |
| Included Risk(s) | 1 | NYCM Insurance |
| Insurance Provided By | 1 | NYCM Insurance |
| LIABILITY LIMIT | 1 | NYCM Insurance |
| Miscellaneous Vehicles | 1 | Plymouth |
| NUMBER | 1 | NYCM Insurance |
| NUMBER OF FAMILIES | 1 | NYCM Insurance |
| Operators | 1 | Plymouth |
| Owner Occupied Properties | 1 | Plymouth |
| Paper Off | 1 | NYCM Insurance |
| Personal Injury | 1 | Plymouth |
| Personal Liability | 1 | NYCM Insurance |
| Personal or Premises Liability | 1 | Plymouth |
| Policy Effective Date | 1 | Plymouth |
| PREMIUM | 1 | NYCM Insurance |
| Protected Since Date | 1 | NYCM Insurance |
| Recreational Vehicle | 1 | NYCM Insurance |
| Rental Properties | 1 | Plymouth |
| Sub County | 1 | NYCM Insurance |
| Table Code | 1 | NYCM Insurance |
| Territory | 1 | NYCM Insurance |
| Transaction Expiration | 1 | NYCM Insurance |
| Transaction Reason Description | 1 | NYCM Insurance |
| TYPE | 1 | NYCM Insurance |
| Type | 1 | NYCM Insurance |
| Uninsured/Underinsured Motorists Liability* | 1 | Plymouth |
| VEHICLES | 1 | NYCM Insurance |
| Vehicles | 1 | Plymouth |
| Watercraft | 1 | NYCM Insurance |
| Watercraft Liability | 1 | Plymouth |
| Watercrafts | 1 | Plymouth |
| YEAR BUILT | 1 | NYCM Insurance |
| Zip | 1 | NYCM Insurance |
