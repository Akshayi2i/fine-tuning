# ocean_marine - core fields (SPEC_21 4.3)

Core field list of the ocean_marine SPEC_21 schema, built from all 4 seed golds of 3 carriers (Markel American Insurance Company, Progressive, Travelers) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `ocean_marine.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 167 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 79 of the 167 |
| Seeds | 4 seed golds, 3 carriers: Markel American Insurance Company, Progressive, Travelers |
| Blocks besides the envelope | `watercraft`, `drivers`, `rating_modifiers` |
| LOB block | none |
| Coverage codes | 22 (`ocean_marine.coverage_codes.yaml`), of them 6 shared `X_` codes |
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
| `document.transaction_type` | Kind of transaction the document records. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `document.transaction_date` | Date the transaction was processed. | - | not printed on the current seeds | 0 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | Change Effective; Policy changes effective | Markel American Insurance Company, Progressive, Travelers | 3 |
| `document.transaction_reason` | Reason for the transaction, as printed. | Reason For Change | Progressive, Travelers | 2 |
| `document.endorsement_number` | Number of the endorsement or policy change. | - | not printed on the current seeds | 0 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `document.issue_date` | Date the document was issued. | Issued On Date | Progressive, Travelers | 3 |
| `document.print_date` | Date the document was printed. | - | Markel American Insurance Company | 1 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | not printed on the current seeds | 0 |
| `carrier.name` | The writing company. | Underwritten by; Your Insurer | Markel American Insurance Company, Progressive, Travelers | 4 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | Travelers | 1 |
| `carrier.address.street` | Street address line. | - | Progressive, Travelers | 2 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.address.city` | City. | - | Markel American Insurance Company, Progressive, Travelers | 3 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | Markel American Insurance Company, Progressive, Travelers | 3 |
| `carrier.address.postal_code` | ZIP code. | - | Progressive, Travelers | 2 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | Progressive | 1 |
| `carrier.fax` | Fax number. | - | not printed on the current seeds | 0 |
| `carrier.web_address` | Website address. | - | Progressive, Travelers | 3 |
| `carrier.claims_phone` | Phone number for reporting a claim. | For Claim Service | Progressive, Travelers | 3 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | Your Agency | Markel American Insurance Company, Progressive, Travelers | 4 |
| `producer.producer_code` | Code the carrier assigns to the agency. | - | Markel American Insurance Company | 1 |
| `producer.contact_name` | Contact person at the agency. | - | not printed on the current seeds | 0 |
| `producer.address.street` | Street address line. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `producer.address.city` | City. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `producer.address.postal_code` | ZIP code. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | For Policy Service | Markel American Insurance Company, Progressive, Travelers | 4 |
| `producer.fax` | Fax number. | - | Travelers | 1 |
| `producer.email` | Email address. | - | not printed on the current seeds | 0 |
| `producer.web_address` | Website of the agency. | - | not printed on the current seeds | 0 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured; Named Insured and Mailing Address | Markel American Insurance Company, Progressive, Travelers | 4 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | Progressive | 2 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | Travelers | 1 |
| `named_insured.mailing_address.city` | City. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | - | not printed on the current seeds | 0 |
| `named_insured.email` | Email address of the insured. | - | Travelers | 1 |
| `named_insured.date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `policy.policy_number` | Policy number. | Policy Number; Policy number; Your Policy Number | Markel American Insurance Company, Progressive, Travelers | 4 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | - | Progressive, Travelers | 3 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | - | Progressive | 1 |
| `policy.effective_date` | Date coverage starts. | Policy Period; Policy Period: From | Markel American Insurance Company, Progressive, Travelers | 4 |
| `policy.expiration_date` | Date coverage ends. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | - | Progressive | 2 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | - | not printed on the current seeds | 0 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | not printed on the current seeds | 0 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | not printed on the current seeds | 0 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `coverages[].limits[].amount` | Amount of the limit. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | Progressive, Travelers | 3 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | Travelers | 1 |
| `coverages[].premium` | Premium for the coverage. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `coverages[].valuation` | Valuation basis. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
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
| `premium.total` | Total premium for the policy. | TOTAL ANNUAL PREMIUM; Total 12 month policy premium; Total Premium; Total Premium for this Policy | Markel American Insurance Company, Progressive, Travelers | 4 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | Minimum Earned Premium | Markel American Insurance Company | 1 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Premium change | Progressive, Travelers | 2 |
| `premium.items[].description` | Description of the premium line as printed. | - | Progressive | 1 |
| `premium.items[].amount` | Premium amount of the line. | - | Progressive | 1 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Markel American Insurance Company | 1 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | Policy Taxes/Fees | Markel American Insurance Company | 1 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Progressive, Travelers | 3 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | Discount if paid in full; Total Savings | Progressive, Travelers | 3 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | - | Progressive | 1 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | not printed on the current seeds | 0 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | not printed on the current seeds | 0 |
| `billing.account_number` | Billing account number the carrier bills under. | Your Account Number | Travelers | 1 |
| `billing.amount_due` | Amount currently due on the bill. | - | Progressive | 1 |
| `billing.due_date` | Date payment is due. | Renewal Payment Due By | Progressive | 1 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | Progressive | 1 |
| `billing.installments[].amount` | Amount of the installment. | - | Progressive | 1 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | Progressive, Travelers | 3 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | Markel American Insurance Company, Travelers | 2 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | not printed on the current seeds | 0 |
| `watercraft[].watercraft_number` | Watercraft number as printed on the schedule. | - | not printed on the current seeds | 0 |
| `watercraft[].hin` | Hull identification number. | HIN; Hull ID #; Hull Identification # | Markel American Insurance Company, Progressive, Travelers | 4 |
| `watercraft[].year` | Model year of the watercraft. | Year | Markel American Insurance Company, Progressive, Travelers | 4 |
| `watercraft[].make` | Manufacturer of the watercraft. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `watercraft[].model` | Model of the watercraft. | - | Markel American Insurance Company, Progressive, Travelers | 4 |
| `watercraft[].length` | Length of the watercraft as printed. | Length | Travelers | 1 |
| `watercraft[].hull_type` | Hull type or material. | - | not printed on the current seeds | 0 |
| `watercraft[].hull_value` | Insured value of the hull. | Agreed Value; Watercraft and Auxiliary Equipment Value | Markel American Insurance Company, Progressive, Travelers | 3 |
| `watercraft[].motors[].year` | Model year of the motor. | - | Progressive | 2 |
| `watercraft[].motors[].make` | Manufacturer of the motor. | Make | Progressive | 2 |
| `watercraft[].motors[].horsepower` | Horsepower of the motor. | Horsepower; Total Horsepower | Progressive | 2 |
| `watercraft[].motors[].serial` | Serial number of the motor. | - | not printed on the current seeds | 0 |
| `watercraft[].navigation_area` | Waters the watercraft may navigate under the policy. | Navigation Limits | Markel American Insurance Company, Progressive, Travelers | 4 |
| `drivers[].driver_number` | Driver number as printed on the schedule. | - | not printed on the current seeds | 0 |
| `drivers[].name` | Driver's name. | Operator Information; Drivers and household residents | Progressive, Travelers | 3 |
| `drivers[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `drivers[].age` | Driver's age as printed. | Age | Progressive | 2 |
| `drivers[].gender` | Gender as printed. | Gender | Progressive | 2 |
| `drivers[].marital_status` | Marital status as printed. | Marital status | Progressive | 2 |
| `drivers[].license_state` | State that issued the driver's licence. | - | not printed on the current seeds | 0 |
| `drivers[].license_number` | Driver's licence number. | - | not printed on the current seeds | 0 |
| `drivers[].relationship` | Relationship to the named insured. | - | Progressive | 2 |
| `drivers[].status` | Driver status (rated, excluded, listed). | - | not printed on the current seeds | 0 |
| `rating_modifiers[].description` | As printed, e.g. 'Experience Modification', 'Multi-Policy Discount'. | - | Progressive, Travelers | 3 |
| `rating_modifiers[].factor` | Modification factor (e.g. 0.85). | - | not printed on the current seeds | 0 |
| `rating_modifiers[].percent` | Modifier as a percentage. | - | not printed on the current seeds | 0 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (1 today: Total Horsepower).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| Total Horsepower | 2 | Progressive, Travelers |
| An installment fee of | 1 | Progressive |
| at | 1 | Markel American Insurance Company |
| Discounts Applied | 1 | Travelers |
| Endorsment Premium | 1 | Markel American Insurance Company |
| Engine Type | 1 | Travelers |
| For Billing Questions | 1 | Travelers |
| Form | 1 | Markel American Insurance Company |
| Includes savings of | 1 | Progressive |
| License Type | 1 | Travelers |
| Make | 1 | Progressive |
| Number of Engines | 1 | Travelers |
| Number of motors | 1 | Progressive |
| PAY IN FULL | 1 | Progressive |
| PAY IN INSTALLMENTS | 1 | Progressive |
| Policy Period | 1 | Markel American Insurance Company |
| Propulsion type | 1 | Progressive |
| Reason For Change | 1 | Travelers |
| Registration number | 1 | Progressive |
| Signed on | 1 | Markel American Insurance Company |
| Summer Mooring Zip | 1 | Travelers |
| Total 12 month policy premium if paid in full | 1 | Progressive |
| Trailer Details | 1 | Travelers |
| Trailer information Year | 1 | Progressive |
| Unit Mooring Location | 1 | Markel American Insurance Company |
| Unit Premium | 1 | Markel American Insurance Company |
| w/Trailer Trip Interruption | 1 | Progressive |
| Watercraft Association | 1 | Travelers |
| Years of Boating Experience | 1 | Travelers |
| Your current policy will expire on | 1 | Progressive |
