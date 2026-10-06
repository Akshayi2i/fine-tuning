# dwelling_fire - core fields (SPEC_21 4.3)

Core field list of the dwelling_fire SPEC_21 schema, built from all 16 seed golds of 4 carriers (Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `dwelling_fire.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 168 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 95 of the 168 |
| Seeds | 16 seed golds, 4 carriers: Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company |
| Blocks besides the envelope | `locations`, `buildings` |
| LOB block | none |
| Coverage codes | 17 (`dwelling_fire.coverage_codes.yaml`), of them 3 shared `X_` codes |
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
| `document.transaction_type` | Kind of transaction the document records. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `document.transaction_date` | Date the transaction was processed. | Process Date | North Country Insurance Company | 2 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | Effective Date of Change; Amended Declarations Page as of | Dryden Mutual, NYCM Insurance | 3 |
| `document.transaction_reason` | Reason for the transaction, as printed. | Transaction Reason Description | Dryden Mutual, NYCM Insurance | 2 |
| `document.endorsement_number` | Number of the endorsement or policy change. | Term-Seq | Dryden Mutual | 1 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `document.issue_date` | Date the document was issued. | - | not printed on the current seeds | 0 |
| `document.print_date` | Date the document was printed. | - | Dryden Mutual, NYCM Insurance | 5 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | not printed on the current seeds | 0 |
| `carrier.name` | The writing company. | Insurance Provided By | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | Dryden Mutual, North Country Insurance Company | 5 |
| `carrier.address.city` | City. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `carrier.address.postal_code` | ZIP code. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | Leatherstocking Cooperative Insurance Company, NYCM Insurance | 11 |
| `carrier.fax` | Fax number. | - | Leatherstocking Cooperative Insurance Company | 9 |
| `carrier.web_address` | Website address. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance | 13 |
| `carrier.claims_phone` | Phone number for reporting a claim. | - | not printed on the current seeds | 0 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `producer.producer_code` | Code the carrier assigns to the agency. | - | Dryden Mutual, NYCM Insurance, North Country Insurance Company | 7 |
| `producer.contact_name` | Contact person at the agency. | - | Dryden Mutual | 1 |
| `producer.address.street` | Street address line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | Leatherstocking Cooperative Insurance Company, NYCM Insurance | 8 |
| `producer.address.city` | City. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `producer.address.postal_code` | ZIP code. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | Phone Number | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `producer.fax` | Fax number. | Fax | Leatherstocking Cooperative Insurance Company, NYCM Insurance | 11 |
| `producer.email` | Email address. | Email Address | Dryden Mutual, NYCM Insurance, North Country Insurance Company | 5 |
| `producer.web_address` | Website of the agency. | - | NYCM Insurance | 2 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured; Named Insured & Mailing Address; Named Insured and Address; Named Insured and Mailing Address | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 7 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.city` | City. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | - | Dryden Mutual | 1 |
| `named_insured.email` | Email address of the insured. | - | Dryden Mutual | 1 |
| `named_insured.date_of_birth` | Date of birth. | DOB | Dryden Mutual | 1 |
| `named_insured.gender` | Gender as printed. | - | NYCM Insurance | 2 |
| `named_insured.marital_status` | Marital status as printed. | - | NYCM Insurance | 2 |
| `named_insured.occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `policy.policy_number` | Policy number. | Policy Number; Policy #; Policy ID | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 15 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance | 14 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | Protected Since Date | NYCM Insurance, North Country Insurance Company | 4 |
| `policy.effective_date` | Date coverage starts. | Effective Date; Policy Term Effective Date; Inception Date | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `policy.expiration_date` | Date coverage ends. | Expiration Date | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | Term Length | NYCM Insurance | 2 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | - | not printed on the current seeds | 0 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | not printed on the current seeds | 0 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 11 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | NYCM Insurance, North Country Insurance Company | 3 |
| `coverages[].limits[].amount` | Amount of the limit. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 14 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | Deductible | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 13 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `coverages[].premium` | Premium for the coverage. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 15 |
| `coverages[].valuation` | Valuation basis. | Loss Settlement; Loss Settlement Building; Loss Settlement Contents | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 14 |
| `coverages[].coinsurance_percent` | Coinsurance percentage that applies to the coverage. | - | not printed on the current seeds | 0 |
| `coverages[].covered_auto_symbols[]` | ISO covered-auto symbols printed against the coverage (1, 2, 7, 8, 9...). Business auto, garage, truckers and package auto parts. | - | not printed on the current seeds | 0 |
| `deductibles[].amount` | Amount of a flat deductible. | Deductible | Dryden Mutual, NYCM Insurance | 3 |
| `deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `interested_parties[].rank` | 1 for first mortgagee, 2 for second... | - | Leatherstocking Cooperative Insurance Company | 4 |
| `interested_parties[].name` | Name of the interested party. | 1st Mortgagee | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 5 |
| `interested_parties[].address.street` | Street address line. | - | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 5 |
| `interested_parties[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.city` | City. | - | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 5 |
| `interested_parties[].address.state` | State, two-letter USPS code in parsed. | - | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 5 |
| `interested_parties[].address.postal_code` | ZIP code. | - | Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 5 |
| `interested_parties[].address.county` | County. | - | not printed on the current seeds | 0 |
| `interested_parties[].loan_number` | Loan or account number with the interested party. | - | not printed on the current seeds | 0 |
| `interested_parties[].is_payor` | True when this party pays the premium (escrow billing). | Payor | North Country Insurance Company | 1 |
| `premium.total` | Total premium for the policy. | Total Annual Policy Premium; Total Policy Premium; Total Quoted Premium; TOTAL PREMIUM | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | - | not printed on the current seeds | 0 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Premium adjustment for this change | Dryden Mutual, NYCM Insurance | 2 |
| `premium.items[].description` | Description of the premium line as printed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance | 6 |
| `premium.items[].amount` | Premium amount of the line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance | 6 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Leatherstocking Cooperative Insurance Company | 9 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | - | Leatherstocking Cooperative Insurance Company | 9 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 10 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company | 9 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | NYCM Insurance, North Country Insurance Company | 3 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | - | NYCM Insurance | 2 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | - | Dryden Mutual | 3 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | Dryden Mutual | 3 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | not printed on the current seeds | 0 |
| `billing.account_number` | Billing account number the carrier bills under. | - | not printed on the current seeds | 0 |
| `billing.amount_due` | Amount currently due on the bill. | - | not printed on the current seeds | 0 |
| `billing.due_date` | Date payment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | not printed on the current seeds | 0 |
| `billing.installments[].amount` | Amount of the installment. | - | not printed on the current seeds | 0 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 15 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, North Country Insurance Company | 13 |
| `locations[].location_number` | Location number as printed on the schedule. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 8 |
| `locations[].address.street` | Street address line. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | Dryden Mutual | 2 |
| `locations[].address.city` | City. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].address.postal_code` | ZIP code. | - | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].address.county` | County. | County; County Name | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].territory` | Rating territory or zone. | Territory | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].protection_class` | Fire protection class of the premises. | Protection Class; Fire Protection | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | Fire District | Dryden Mutual, NYCM Insurance | 5 |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | not printed on the current seeds | 0 |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | Miles From Fire Dept | Dryden Mutual | 3 |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | Feet From Hydrant | Dryden Mutual | 3 |
| `buildings[].building_number` | Building number as printed on the schedule. | - | Dryden Mutual, North Country Insurance Company | 4 |
| `buildings[].construction` | Construction type (frame, masonry, fire resistive...). | Construction; Construction Type | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `buildings[].occupancy` | How the building is occupied or used. | Occupancy; Risk Description | Dryden Mutual, Leatherstocking Cooperative Insurance Company, NYCM Insurance, North Country Insurance Company | 16 |
| `buildings[].year_built` | Year the building was built. | Year Built; Year of Construction | Dryden Mutual, NYCM Insurance, North Country Insurance Company | 7 |
| `buildings[].area_sq_ft` | Floor area in square feet. | - | not printed on the current seeds | 0 |
| `buildings[].stories` | Number of stories. | - | not printed on the current seeds | 0 |
| `buildings[].number_of_units` | Number of dwelling units or families in the building. | # of Units | NYCM Insurance | 2 |
| `buildings[].roof_type` | Roof type or covering material. | Roof Type | Dryden Mutual | 1 |
| `buildings[].protective_devices[]` | Protective devices fitted (alarms, sprinklers, extinguishers...). | - | not printed on the current seeds | 0 |
| `buildings[].basement` | Whether the building has a basement. | - | not printed on the current seeds | 0 |
| `buildings[].sprinklered` | Whether the building has an automatic sprinkler system. | - | not printed on the current seeds | 0 |
| `buildings[].heating_type` | Type of the primary heating system. | - | not printed on the current seeds | 0 |
| `buildings[].heating_fuel` | Fuel of the primary heating system. | - | not printed on the current seeds | 0 |
| `buildings[].heating_year_updated` | Year the primary heating system was last updated. | - | not printed on the current seeds | 0 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (0 today).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| # of Families | 1 | North Country Insurance Company |
| 1st Mortgagee | 1 | Leatherstocking Cooperative Insurance Company |
| Agency Address | 1 | NYCM Insurance |
| AGENCY CUSTOMER ID | 1 | Dryden Mutual |
| Agency Territory | 1 | NYCM Insurance |
| Assigned To | 1 | North Country Insurance Company |
| Auto Increase in Insurance | 1 | Dryden Mutual |
| BASIC FORM | 1 | NYCM Insurance |
| Basic Form | 1 | NYCM Insurance |
| Basic Premium | 1 | North Country Insurance Company |
| Billing | 1 | NYCM Insurance |
| Billing Information | 1 | Dryden Mutual |
| Building Theft Coverage | 1 | Dryden Mutual |
| Burglar Alarm | 1 | North Country Insurance Company |
| CANCELLATION REQUEST / POLICY RELEASE | 1 | Dryden Mutual |
| Cause of Loss Form | 1 | Dryden Mutual |
| Change In Annual Premium | 1 | Dryden Mutual |
| Classification | 1 | Leatherstocking Cooperative Insurance Company |
| Color | 1 | Leatherstocking Cooperative Insurance Company |
| Condition | 1 | Leatherstocking Cooperative Insurance Company |
| Copy | 1 | North Country Insurance Company |
| County | 1 | North Country Insurance Company |
| County Code | 1 | NYCM Insurance |
| Coverage Premium | 1 | Leatherstocking Cooperative Insurance Company |
| Date | 1 | Dryden Mutual |
| Deductible | 1 | North Country Insurance Company |
| Deductible Credit | 1 | North Country Insurance Company |
| Description | 1 | North Country Insurance Company |
| DESCRIPTION | 1 | NYCM Insurance |
| Dimensions | 1 | Leatherstocking Cooperative Insurance Company |
| Direct Mail | 1 | NYCM Insurance |
| DOB | 1 | Dryden Mutual |
| Document History | 1 | Dryden Mutual |
| Document Reference | 1 | Dryden Mutual |
| Document Region | 1 | Dryden Mutual |
| Document Title | 1 | Dryden Mutual |
| Dwelling Package VIP Plus (FL2 and FL3) | 1 | Dryden Mutual |
| Each Person | 1 | Dryden Mutual |
| Expiration Date | 1 | North Country Insurance Company |
| Families | 1 | Dryden Mutual |
| File # | 1 | Dryden Mutual |
| FL-1 BASIC FORM Premium | 1 | NYCM Insurance |
| Forms/Endorsements Premium | 1 | North Country Insurance Company |
| INFLATION GUARD | 1 | NYCM Insurance |
| Inland Marine Premium | 1 | Dryden Mutual |
| Insured Name | 1 | Dryden Mutual |
| INSURED NAME AND ADDRESS | 1 | Dryden Mutual |
| Insured Type | 1 | NYCM Insurance |
| Interest Type | 1 | North Country Insurance Company |
| Lead Abatement | 1 | NYCM Insurance |
| Limit | 1 | Dryden Mutual |
| Limit: Each Occurrence | 1 | Dryden Mutual |
| Loc #/Bldg # | 1 | North Country Insurance Company |
| Location Description | 1 | NYCM Insurance |
| Location on Premises | 1 | Leatherstocking Cooperative Insurance Company |
| Loss Settlement Building | 1 | Dryden Mutual |
| Mail To | 1 | Leatherstocking Cooperative Insurance Company |
| Market Value | 1 | Dryden Mutual |
| Modifies Coverage(s) at Renewal | 1 | Leatherstocking Cooperative Insurance Company |
| Mortgagee Information | 1 | Dryden Mutual |
| Name of Applicant | 1 | Dryden Mutual |
| Paper Off | 1 | NYCM Insurance |
| Participants | 1 | Dryden Mutual |
| Phone | 1 | Dryden Mutual |
| Please provide Policy Number(s) | 1 | Dryden Mutual |
| POLICY NUMBER | 1 | Dryden Mutual |
| POLICY NUMBER EFFECTIVE DATE APPLICANT / NAMED INSURED(S) | 1 | Dryden Mutual |
| Policy Period | 1 | Dryden Mutual |
| POLICY TERM | 1 | Dryden Mutual |
| Policy Term Effective Date | 1 | Leatherstocking Cooperative Insurance Company |
| Policy Term Expiration Date | 1 | Leatherstocking Cooperative Insurance Company |
| Previous Insurance Carrier | 1 | Dryden Mutual |
| Prior Annual Premium | 1 | Dryden Mutual |
| Property | 1 | Leatherstocking Cooperative Insurance Company |
| Property Total | 1 | Leatherstocking Cooperative Insurance Company |
| Protected Since Date | 1 | NYCM Insurance |
| Protective Devices | 1 | Dryden Mutual |
| REASON FOR CANCELLATION | 1 | Dryden Mutual |
| Renovator Credit | 1 | Dryden Mutual |
| Seasonal | 1 | NYCM Insurance |
| Sender Email | 1 | Dryden Mutual |
| Special Rating Conditions | 1 | Dryden Mutual |
| Sprinklers | 1 | North Country Insurance Company |
| Structure | 1 | North Country Insurance Company |
| Surcharge Information | 1 | NYCM Insurance |
| Term-Seq | 1 | Dryden Mutual |
| Territory | 1 | NYCM Insurance |
| Total Annual Policy Premium | 1 | Dryden Mutual |
| Total Annual Premium for Location # 1 | 1 | Dryden Mutual |
| Total Annual Premium for Location # 2 | 1 | Dryden Mutual |
| Total Annual Premium This Location: | 1 | Dryden Mutual |
| TOTAL LOCATION PREMIUM | 1 | NYCM Insurance |
| Total Number of Risks | 1 | Dryden Mutual |
| Total Quoted Premium | 1 | Dryden Mutual |
| Transaction Expiration | 1 | NYCM Insurance |
| Transaction Reason Description | 1 | NYCM Insurance |
| Type | 1 | Dryden Mutual |
| TYPE | 1 | NYCM Insurance |
| Type of Building | 1 | Leatherstocking Cooperative Insurance Company |
| Type of Business | 1 | Leatherstocking Cooperative Insurance Company |
| Type of Device | 1 | Leatherstocking Cooperative Insurance Company |
| Unoccupied/Seasonal Surcharge | 1 | North Country Insurance Company |
| US/Eastern | 1 | Dryden Mutual |
| Usage | 1 | North Country Insurance Company |
| Zone | 1 | Leatherstocking Cooperative Insurance Company |

## Notes

- The engine-test gold of the Dryden Mutual dwelling application used the homeowners codes `HO_COV_A/B/C`; the canonical codes are `DF_COV_A/B/C` (see CHANGELOG).
