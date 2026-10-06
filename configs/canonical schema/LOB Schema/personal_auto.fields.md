# personal_auto - core fields (SPEC_21 4.3)

Core field list of the personal_auto SPEC_21 schema, built from all 20 seed golds of 9 carriers (AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `personal_auto.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 180 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 106 of the 180 |
| Seeds | 20 seed golds, 9 carriers: AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers |
| Blocks besides the envelope | `locations`, `vehicles`, `drivers`, `rating_modifiers` |
| LOB block | none |
| Coverage codes | 33 (`personal_auto.coverage_codes.yaml`), of them 9 shared `X_` codes |
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
| `document.transaction_type` | Kind of transaction the document records. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, NYCM Insurance, Plymouth, Progressive, Travelers | 18 |
| `document.transaction_date` | Date the transaction was processed. | TRANSACTION DATE | MAPFRE Insurance Company | 1 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | TRANSACTION EFFECTIVE DATE; POLICY CHANGES MADE AS OF | AEIC, Hagerty Insurance, MAPFRE Insurance Company, NYCM Insurance, Plymouth, Travelers | 12 |
| `document.transaction_reason` | Reason for the transaction, as printed. | Transaction Reason Description; SUMMARY OF CHANGES | AEIC, NYCM Insurance, Plymouth, Travelers | 6 |
| `document.endorsement_number` | Number of the endorsement or policy change. | This is change number | Travelers | 1 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `document.issue_date` | Date the document was issued. | Issue Date; Issued On; Issued On Date | Hagerty Insurance, Plymouth, Travelers | 9 |
| `document.print_date` | Date the document was printed. | PRINT DATE; Date Summary Printed | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, NYCM Insurance | 13 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | Date Mailed; Date of Mailing | Mercury Insurance Company, Progressive | 2 |
| `carrier.name` | The writing company. | Insurance Provided By; Issued By (Name of Insurance Company); Policy Issued by; Underwritten By; Underwritten by; Your Insurer | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Travelers | 17 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.address.city` | City. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Travelers | 17 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Travelers | 17 |
| `carrier.address.postal_code` | ZIP code. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Travelers | 17 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | NYCM Insurance, Progressive | 7 |
| `carrier.fax` | Fax number. | - | Travelers | 1 |
| `carrier.web_address` | Website address. | - | NYCM Insurance, Plymouth | 8 |
| `carrier.claims_phone` | Phone number for reporting a claim. | 24-hour claim service; Service and Claims; To Report a Claim; To report a claim please call | Hagerty Insurance, Mercury Insurance Company, Plymouth, Progressive, Travelers | 10 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | Agent Name and Address; YOUR AGENT IS; Your agency is | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `producer.producer_code` | Code the carrier assigns to the agency. | AGENCY CODE; AGENCY ID; PRODUCER CODE | AEIC, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Travelers | 10 |
| `producer.contact_name` | Contact person at the agency. | - | not printed on the current seeds | 0 |
| `producer.address.street` | Street address line. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 18 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | Mercury Insurance Company, NYCM Insurance | 4 |
| `producer.address.city` | City. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 18 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 18 |
| `producer.address.postal_code` | ZIP code. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 18 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | Agent Telephone Number; For Policy Service; For Policy Service Call | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `producer.fax` | Fax number. | - | NYCM Insurance | 6 |
| `producer.email` | Email address. | - | NYCM Insurance | 6 |
| `producer.web_address` | Website of the agency. | - | NYCM Insurance | 2 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured; NAMED INSURED; 1. Named Insured; APPLICANT NAME AND ADDRESS; INSURED NAME | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 11 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.city` | City. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | Home Phone | MAPFRE Insurance Company, NYCM Insurance | 5 |
| `named_insured.email` | Email address of the insured. | Email Address | MAPFRE Insurance Company, NYCM Insurance | 5 |
| `named_insured.date_of_birth` | Date of birth. | Date of Birth | MAPFRE Insurance Company | 1 |
| `named_insured.gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `policy.policy_number` | Policy number. | Policy Number; POLICY NUMBER; Policy #; Your Policy Number; POLICY # | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | POLICY TYPE | Foremost Insurance Company, MAPFRE Insurance Company, NYCM Insurance | 4 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | INCEPTION DATE; Inception Date; Protected Since | MAPFRE Insurance Company, NYCM Insurance, Progressive | 4 |
| `policy.effective_date` | Date coverage starts. | Effective Date; Policy Period; EFFECTIVE DATE | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `policy.expiration_date` | Date coverage ends. | Expiration Date; EXPIRATION DATE; Policy Expiration Date | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | Term Length; TERM | MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Progressive | 5 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | Policy State | Hagerty Insurance | 4 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | AEIC | 1 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | not printed on the current seeds | 0 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `coverages[].limits[].amount` | Amount of the limit. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 17 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | Deductible | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | Hagerty Insurance, Progressive, Travelers | 6 |
| `coverages[].premium` | Premium for the coverage. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 15 |
| `coverages[].valuation` | Valuation basis. | Actual Cash Value (ACV) or Limit | Foremost Insurance Company, Mercury Insurance Company, Progressive, Travelers | 5 |
| `coverages[].coinsurance_percent` | Coinsurance percentage that applies to the coverage. | - | not printed on the current seeds | 0 |
| `coverages[].covered_auto_symbols[]` | ISO covered-auto symbols printed against the coverage (1, 2, 7, 8, 9...). Business auto, garage, truckers and package auto parts. | - | not printed on the current seeds | 0 |
| `deductibles[].amount` | Amount of a flat deductible. | - | not printed on the current seeds | 0 |
| `deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `interested_parties[].rank` | 1 for first mortgagee, 2 for second... | - | not printed on the current seeds | 0 |
| `interested_parties[].name` | Name of the interested party. | - | AEIC, NYCM Insurance, Plymouth, Progressive, Travelers | 11 |
| `interested_parties[].address.street` | Street address line. | - | AEIC, NYCM Insurance, Travelers | 8 |
| `interested_parties[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `interested_parties[].address.city` | City. | - | AEIC, NYCM Insurance, Progressive, Travelers | 9 |
| `interested_parties[].address.state` | State, two-letter USPS code in parsed. | - | AEIC, NYCM Insurance, Progressive, Travelers | 9 |
| `interested_parties[].address.postal_code` | ZIP code. | - | AEIC, NYCM Insurance, Progressive, Travelers | 9 |
| `interested_parties[].address.county` | County. | - | not printed on the current seeds | 0 |
| `interested_parties[].loan_number` | Loan or account number with the interested party. | - | NYCM Insurance | 2 |
| `interested_parties[].is_payor` | True when this party pays the premium (escrow billing). | - | not printed on the current seeds | 0 |
| `premium.total` | Total premium for the policy. | TOTAL PREMIUM; Total Policy Premium; Total Annual Premium; Total Premium for this Policy; TERM AMOUNT; Total 6 Month Policy Premium (All Vehicles); Total 6 month policy premium and fees | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 16 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | - | not printed on the current seeds | 0 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Quoted Pro-rated Premium Amount; Total return premium | AEIC, NYCM Insurance, Plymouth, Travelers | 5 |
| `premium.items[].description` | Description of the premium line as printed. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `premium.items[].amount` | Premium amount of the line. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, MAPFRE Insurance Company, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 20 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, Plymouth, Progressive, Travelers | 11 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | MVLE Fee; Motor Vehicle Law Enforcement Fee; Mandatory New York Law Enforcement Fee; Motor Vehicle Law Enforcement Fee (MVLE Fee); Motor vehicle law enforcement fee | AEIC, Hagerty Insurance, Mercury Insurance Company, Plymouth, Progressive, Travelers | 11 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Plymouth | 1 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | Plymouth | 1 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Plymouth, Progressive, Travelers | 3 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | - | Plymouth, Progressive, Travelers | 3 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | - | Plymouth | 3 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | not printed on the current seeds | 0 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | - | not printed on the current seeds | 0 |
| `billing.account_number` | Billing account number the carrier bills under. | Your Account Number | Travelers | 1 |
| `billing.amount_due` | Amount currently due on the bill. | - | not printed on the current seeds | 0 |
| `billing.due_date` | Date payment is due. | Renewal Payment Due By | Progressive | 1 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | Progressive | 1 |
| `billing.installments[].amount` | Amount of the installment. | - | Progressive | 1 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 16 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 14 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | AEIC, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 15 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | not printed on the current seeds | 0 |
| `locations[].location_number` | Location number as printed on the schedule. | - | not printed on the current seeds | 0 |
| `locations[].address.street` | Street address line. | - | MAPFRE Insurance Company, NYCM Insurance | 4 |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `locations[].address.city` | City. | - | MAPFRE Insurance Company, NYCM Insurance, Travelers | 5 |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | MAPFRE Insurance Company, NYCM Insurance, Travelers | 5 |
| `locations[].address.postal_code` | ZIP code. | Garaging ZIP Code | MAPFRE Insurance Company, NYCM Insurance, Progressive | 5 |
| `locations[].address.county` | County. | County Name | NYCM Insurance | 2 |
| `locations[].territory` | Rating territory or zone. | Territory | AEIC, Hagerty Insurance, NYCM Insurance, Plymouth, Travelers | 8 |
| `locations[].protection_class` | Fire protection class of the premises. | - | not printed on the current seeds | 0 |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | - | not printed on the current seeds | 0 |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | not printed on the current seeds | 0 |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | - | not printed on the current seeds | 0 |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | - | not printed on the current seeds | 0 |
| `vehicles[].vehicle_number` | Vehicle number as printed on the schedule. | VEH; VEHICLE | AEIC, Foremost Insurance Company, Hagerty Insurance, NYCM Insurance, Travelers | 14 |
| `vehicles[].year` | Model year. | Year | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `vehicles[].make` | Manufacturer. | Make | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `vehicles[].model` | Model. | - | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `vehicles[].vin` | Vehicle identification number. | VIN | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `vehicles[].body_type` | A trailer is a vehicle with body_type 'Trailer'. | Vehicle Type | AEIC, Foremost Insurance Company, Hagerty Insurance, Progressive | 7 |
| `vehicles[].use` | How the vehicle is used (pleasure, commute, business...). | Primary use of the vehicle | AEIC, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 6 |
| `vehicles[].annual_mileage` | As printed; often a band ('10,000 - 11,999'). | Annual miles | Progressive, Travelers | 2 |
| `vehicles[].commute_miles_one_way` | One-way commuting distance printed for the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].engine_displacement` | Engine size as printed (e.g. '1,745 cc'). | - | not printed on the current seeds | 0 |
| `vehicles[].cost_new` | Original cost new of the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].stated_amount` | Stated or agreed value of the vehicle. | AGREED VALUE; Guaranteed Value | AEIC, Hagerty Insurance | 5 |
| `vehicles[].symbols[]` | Rating symbols printed for the vehicle. | BI SYMBOL; COLL SYMBOL; CSL SYMBOL; MP SYMBOL; OTC SYMBOL | AEIC, NYCM Insurance, Plymouth | 5 |
| `drivers[].driver_number` | Driver number as printed on the schedule. | - | NYCM Insurance, Travelers | 5 |
| `drivers[].name` | Driver's name. | Driver(s); Drivers; Drivers and household residents | AEIC, Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 19 |
| `drivers[].date_of_birth` | Date of birth. | - | Foremost Insurance Company, Hagerty Insurance, Mercury Insurance Company, NYCM Insurance, Plymouth, Travelers | 15 |
| `drivers[].age` | Driver's age as printed. | Age | Plymouth, Progressive | 2 |
| `drivers[].gender` | Gender as printed. | Gender | AEIC, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 11 |
| `drivers[].marital_status` | Marital status as printed. | Marital status | AEIC, Mercury Insurance Company, NYCM Insurance, Plymouth, Progressive, Travelers | 12 |
| `drivers[].license_state` | State that issued the driver's licence. | - | AEIC | 1 |
| `drivers[].license_number` | Driver's licence number. | - | NYCM Insurance | 4 |
| `drivers[].relationship` | Relationship to the named insured. | - | Foremost Insurance Company, Mercury Insurance Company, NYCM Insurance, Progressive | 5 |
| `drivers[].status` | Driver status (rated, excluded, listed). | - | NYCM Insurance, Plymouth, Travelers | 6 |
| `rating_modifiers[].description` | As printed, e.g. 'Experience Modification', 'Multi-Policy Discount'. | - | AEIC, Hagerty Insurance, NYCM Insurance, Plymouth | 11 |
| `rating_modifiers[].factor` | Modification factor (e.g. 0.85). | - | not printed on the current seeds | 0 |
| `rating_modifiers[].percent` | Modifier as a percentage. | - | AEIC, NYCM Insurance | 5 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (17 today: CLASS; Insured's bodily injury damage claim paid to spouse; Insured's bodily injury policy coverage limit; Insured’s Bodily Injury Damages; Insured’s Combined Single Liability (CSL) Limit; Insured’s Liability Limit; Insured’s SUM Limit; Maximum; Minimum; POLICY NUMBER; Policy Period; and; own in excess of; penalty and; st; to; were both).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| Maximum | 3 | Foremost Insurance Company, NYCM Insurance, Travelers |
| Minimum | 3 | Foremost Insurance Company, NYCM Insurance, Travelers |
| and | 2 | NYCM Insurance, Travelers |
| CLASS | 2 | AEIC, NYCM Insurance |
| Insured's bodily injury damage claim paid to spouse | 2 | Foremost Insurance Company, Travelers |
| Insured's bodily injury policy coverage limit | 2 | Foremost Insurance Company, Travelers |
| Insured’s Bodily Injury Damages | 2 | Foremost Insurance Company, NYCM Insurance |
| Insured’s Combined Single Liability (CSL) Limit | 2 | NYCM Insurance, Travelers |
| Insured’s Liability Limit | 2 | Foremost Insurance Company, NYCM Insurance |
| Insured’s SUM Limit | 2 | Foremost Insurance Company, NYCM Insurance |
| own in excess of | 2 | NYCM Insurance, Travelers |
| penalty and | 2 | Foremost Insurance Company, NYCM Insurance |
| POLICY NUMBER | 2 | AEIC, NYCM Insurance |
| Policy Period | 2 | Mercury Insurance Company, Travelers |
| st | 2 | Foremost Insurance Company, NYCM Insurance |
| to | 2 | Foremost Insurance Company, NYCM Insurance |
| were both | 2 | Foremost Insurance Company, NYCM Insurance |
| ($5.00 per vehicle semi-annually) | 1 | Travelers |
| (no printed label) | 1 | Foremost Insurance Company |
| . Insured recovers | 1 | NYCM Insurance |
| 100 fee to | 1 | Foremost Insurance Company |
| 174.00 Multiple Cars | 1 | Travelers |
| 1st Offense | 1 | Travelers |
| 2. 80% of lost earnings up to a maximum monthly payment of | 1 | Foremost Insurance Company |
| 25,000 Limit, $1,000 Deductible | 1 | Foremost Insurance Company |
| 25,000 per injured person and, subject to this per person limit | 1 | Foremost Insurance Company |
| 275,000) are less than the $300,000 CSL SUM limit. | 1 | Travelers |
| 3. if the covered automobile is stolen, we will pay up to | 1 | Foremost Insurance Company |
| 3. up to | 1 | Foremost Insurance Company |
| 750 DRA, as applicable. | 1 | Foremost Insurance Company |
| 800-922-FRAUD | 1 | Foremost Insurance Company |
| 9-3 ARC WRANGLER U GLADIATOR | 1 | Travelers |
| A 7.40-7.50*4.50MM. APPROX WT 1.52CT H COL AND SI2 CLR | 1 | Foremost Insurance Company |
| a benefit, up to a maximum of | 1 | Foremost Insurance Company |
| a CDL) and must pay a civil penalty of | 1 | NYCM Insurance |
| ACCIDENT PREVENTION COURSE | 1 | NYCM Insurance |
| Additional Household Members (List Only) | 1 | Mercury Insurance Company |
| Additional Interests | 1 | Hagerty Insurance |
| AGENCY CODE | 1 | NYCM Insurance |
| Aggregate No Fault Benefits Available | 1 | Travelers |
| AL3 Viewer - HawkSoft | 1 | MAPFRE Insurance Company |
| also will be charged a | 1 | Foremost Insurance Company |
| also will be charged a DRA payable in three annual payments of | 1 | Foremost Insurance Company |
| Amount | 1 | Foremost Insurance Company |
| Another Passenger’s Damages that resulted in death | 1 | NYCM Insurance |
| Anti-Lock Brake | 1 | Mercury Insurance Company |
| ANTI-LOCK BRAKES | 1 | NYCM Insurance |
| Anti-lock Braking System discount | 1 | Plymouth |
| Anti-Theft | 1 | Mercury Insurance Company |
| ANTI-THEFT DEVICE | 1 | NYCM Insurance |
| APCD | 1 | NYCM Insurance |
| APCD DATE INFO | 1 | NYCM Insurance |
| As of , your Deductible Savings Benefit is | 1 | Foremost Insurance Company |
| AUTOMOBILE | 1 | Foremost Insurance Company |
| AUTOMOBILE: / BODILY INJURY AND | 1 | Foremost Insurance Company |
| Back Up of Sewer, Drain and Sump Pump Coverage | 1 | Foremost Insurance Company |
| BAND | 1 | AEIC |
| BANK, WE WILL CHARGE YOU A | 1 | Foremost Insurance Company |
| BASIC PIP | 1 | Foremost Insurance Company |
| BASIC PIP /PERSON | 1 | Foremost Insurance Company |
| be | 1 | Travelers |
| Bill by Mail / Email | 1 | Travelers |
| Billing and Payment Information. . . | 1 | Travelers |
| Bodily Injury | 1 | Foremost Insurance Company |
| Bodily Injury $ 250,000 Each Person | 1 | Foremost Insurance Company |
| Bodily Injury and | 1 | Foremost Insurance Company |
| Bodily Injury Per | 1 | Foremost Insurance Company |
| Bodily Injury Per Person | 1 | Foremost Insurance Company |
| BRILL CUT DIA APPROX WT 1.21CT G-H COL, SI1 | 1 | Foremost Insurance Company |
| c. where the amount of the settlement exceeds | 1 | Foremost Insurance Company |
| CANCELLATION FOR NONPAYMENT IS ISSUED, A | 1 | Foremost Insurance Company |
| CHARGEABLE ACCIDENT ON | 1 | Foremost Insurance Company |
| CHARGEABLE MORE OR EQUAL | 1 | NYCM Insurance |
| charges a | 1 | Foremost Insurance Company |
| Chemical Test | 1 | NYCM Insurance |
| claimants, subject to a maximum of per person | 1 | Foremost Insurance Company |
| CLAIMS FREE DISCOUNT | 1 | NYCM Insurance |
| Class Code | 1 | Travelers |
| Class Item Insurance Description | 1 | Foremost Insurance Company |
| Client Number | 1 | Hagerty Insurance |
| Co-Insured Information | 1 | MAPFRE Insurance Company |
| CODED SCORE | 1 | NYCM Insurance |
| COLL (NOT AT FAULT MORE THAN | 1 | NYCM Insurance |
| Collision coverages totaling more than | 1 | Foremost Insurance Company |
| Collision Less Deductible | 1 | Foremost Insurance Company |
| Comprehensive | 1 | Mercury Insurance Company |
| Contact Info | 1 | MAPFRE Insurance Company |
| Continuous Insurance | 1 | Mercury Insurance Company |
| COUPLER (COMBO) - BODILY INJURY | 1 | NYCM Insurance |
| COUPLER (COMBO) - COLLISION | 1 | NYCM Insurance |
| COUPLER (COMBO) - FULL GLASS | 1 | NYCM Insurance |
| COUPLER (COMBO) - OTHER THAN COLLISION | 1 | NYCM Insurance |
| coverage that will pay certain expenses, up to | 1 | Foremost Insurance Company |
| CTR within 5 | 1 | NYCM Insurance |
| CTR-under | 1 | NYCM Insurance |
| Current Annual Premium | 1 | Foremost Insurance Company |
| Current Coverage A - Dwelling amount with Inflation Factor | 1 | Foremost Insurance Company |
| DATE | 1 | NYCM Insurance |
| DATE FIRST LICENSED | 1 | AEIC |
| Date of Birth | 1 | MAPFRE Insurance Company |
| Day | 1 | Foremost Insurance Company |
| DAYS PER | 1 | NYCM Insurance |
| Daytime Running Lamps | 1 | Mercury Insurance Company |
| DAYTIME RUNNING LAMPS | 1 | NYCM Insurance |
| Death Benefit | 1 | Travelers |
| Deductible | 1 | Foremost Insurance Company |
| Deductible Savings Benefit (DSB) | 1 | Foremost Insurance Company |
| DESCRIPTION | 1 | NYCM Insurance |
| Direct Mail | 1 | NYCM Insurance |
| Discount if paid in full | 1 | Progressive |
| Discounts Included in Your Premium | 1 | Travelers |
| DMV FEE | 1 | NYCM Insurance |
| DOB | 1 | NYCM Insurance |
| driver's license by DMV. You also will be charged a | 1 | NYCM Insurance |
| driving-online | 1 | NYCM Insurance |
| each accident basis | 1 | Foremost Insurance Company |
| Each Occurrence | 1 | Foremost Insurance Company |
| Each Person | 1 | Foremost Insurance Company |
| each person | 1 | Travelers |
| ELECTRONIC FUND TRANSFER | 1 | NYCM Insurance |
| Electronic Funds Transfer (EFT) | 1 | Travelers |
| email. Other charges that may apply include a | 1 | Travelers |
| email. Other charges that may apply include a late charge and a | 1 | Travelers |
| Enforcement Fee | 1 | Hagerty Insurance |
| enrollment, and the DDP charges an additional course fee of up to | 1 | NYCM Insurance |
| eSignature | 1 | Mercury Insurance Company |
| Excluded Person(s) | 1 | Hagerty Insurance |
| FAX | 1 | Foremost Insurance Company |
| fees, and meet other eligibility requirements. DMV charges a | 1 | NYCM Insurance |
| Felony) | 1 | NYCM Insurance |
| Final Market | 1 | Mercury Insurance Company |
| For 24-hour towing/roadside assistance, or Claims, call | 1 | Foremost Insurance Company |
| for comprehensive loss and | 1 | Foremost Insurance Company |
| from the negligent owner or operator of the other motor vehicle, and | 1 | Foremost Insurance Company |
| Garaging | 1 | Hagerty Insurance |
| Garaging ZIP Code | 1 | Mercury Insurance Company |
| Gender | 1 | MAPFRE Insurance Company |
| GENERAL POLICY INFORMATION | 1 | NYCM Insurance |
| HOME OWNERSHIP | 1 | NYCM Insurance |
| HOMEOWNERSHIP | 1 | NYCM Insurance |
| If the insured’s CSL and CSL SUM limit were each | 1 | NYCM Insurance |
| If your CSL and CSL SUM limit were each | 1 | Travelers |
| If your CSL and CSL SUM limit were each and your damages amounted to | 1 | Travelers |
| If your policy provides Physical Damage coverage, a | 1 | Travelers |
| in excess of | 1 | Foremost Insurance Company |
| Incident Date | 1 | Plymouth |
| Includes savings of | 1 | Progressive |
| inexperienced operator(s), which total | 1 | Plymouth |
| Infraction) | 1 | NYCM Insurance |
| INSTALLMENT CHARGE | 1 | AEIC |
| installment fee | 1 | Progressive |
| Insurance Provided By | 1 | NYCM Insurance |
| insurance until the | 1 | Foremost Insurance Company |
| insurance until the , or | 1 | Foremost Insurance Company |
| Insured's Bodily Injury Damages | 1 | Travelers |
| Insured’s bodily injury damage claim paid to spouse | 1 | NYCM Insurance |
| Insured’s bodily injury damage claim paid to spouse: | 1 | Travelers |
| Insured’s bodily injury policy coverage limit | 1 | NYCM Insurance |
| INSURED’S RETAINED LIMIT | 1 | Foremost Insurance Company |
| ISSUE DATE | 1 | NYCM Insurance |
| Issued on | 1 | Travelers |
| Jewelry-In-Vault Premium | 1 | Foremost Insurance Company |
| Jewelry-In-Vault Premium $ 84.00 | 1 | Foremost Insurance Company |
| Jewelry-Out-of-Vault Premium | 1 | Foremost Insurance Company |
| Law (ZTL) | 1 | NYCM Insurance |
| LIABILITY | 1 | Foremost Insurance Company |
| liability and SUM limits of | 1 | NYCM Insurance |
| liability and SUM limits of or more, the SUM recovery would then be | 1 | NYCM Insurance |
| License Status | 1 | Mercury Insurance Company |
| Lienholder/Lease Company | 1 | Plymouth |
| Limit | 1 | Foremost Insurance Company |
| limit for each person, we will provide | 1 | Foremost Insurance Company |
| LOCATION | 1 | MAPFRE Insurance Company |
| Locations | 1 | MAPFRE Insurance Company |
| Mandatory Basic Economic Loss | 1 | Travelers |
| Marital Status | 1 | MAPFRE Insurance Company |
| Maximum Monthly Work Loss | 1 | Travelers |
| maximum of per person | 1 | Travelers |
| MILES PER | 1 | NYCM Insurance |
| Misdemeanor) | 1 | NYCM Insurance |
| more within 10 | 1 | NYCM Insurance |
| Motor Vehicle Law Enforcement Fee (state mandated) | 1 | Hagerty Insurance |
| MULTI-CAR | 1 | NYCM Insurance |
| Multi-Policy | 1 | Mercury Insurance Company |
| Multi-Vehicle Discount | 1 | Hagerty Insurance |
| Multiple Cars | 1 | Travelers |
| Musical Instrument Premium | 1 | Foremost Insurance Company |
| must pay a civil penalty of | 1 | Travelers |
| NAME | 1 | NYCM Insurance |
| NAME AND ADDRESS | 1 | NYCM Insurance |
| NAME/ADDRESS | 1 | NYCM Insurance |
| NAMED INSURED(S) | 1 | MAPFRE Insurance Company |
| NEW CAR DISCOUNT | 1 | NYCM Insurance |
| NEW YORK MOTOR VEHICLE LAW ENFORCEMENT FEE: | 1 | Foremost Insurance Company |
| NONPAYMENT IS ISSUED, A | 1 | Foremost Insurance Company |
| NSF BY A BANK, WE WILL CHARGE YOU A | 1 | Foremost Insurance Company |
| OBEL | 1 | Foremost Insurance Company |
| Occupation | 1 | MAPFRE Insurance Company |
| of $100,000 per person: | 1 | Travelers |
| of any single accident shall not exceed | 1 | Foremost Insurance Company |
| of per person | 1 | NYCM Insurance |
| of your future anniversary dates, until the maximum policy benefit of | 1 | Foremost Insurance Company |
| offenses, then you will be required to pay a civil penalty of | 1 | NYCM Insurance |
| OL AND VS2 CLR. WT OF RING OF DIA 4.9 GRMAS | 1 | Foremost Insurance Company |
| on | 1 | Mercury Insurance Company |
| ORDERED DATE | 1 | NYCM Insurance |
| ORIGINAL COST NEW | 1 | NYCM Insurance |
| Other Motor Vehicle Liability Limit | 1 | NYCM Insurance |
| Other Motor Vehicle’s Liability Limit | 1 | Foremost Insurance Company |
| Other Necessary Expenses | 1 | Foremost Insurance Company |
| Other Necessary Expenses (per day) | 1 | Travelers |
| PACKAGE POLICY NUMBER | 1 | AEIC |
| Paid in Full Discount | 1 | Hagerty Insurance |
| PAID IN FULL DISCOUNT | 1 | NYCM Insurance |
| Paper Off | 1 | NYCM Insurance |
| Passive Restraint | 1 | Mercury Insurance Company |
| PASSIVE RESTRAINT | 1 | NYCM Insurance |
| Passive Restraint discount | 1 | Plymouth |
| PAY IN INSTALLMENTS | 1 | Progressive |
| per month and | 1 | Foremost Insurance Company |
| PER PERSON | 1 | Foremost Insurance Company |
| per person | 1 | Travelers |
| PERSON | 1 | Foremost Insurance Company |
| person and | 1 | NYCM Insurance |
| person and per accident for injured persons and per person and | 1 | NYCM Insurance |
| person limit | 1 | Foremost Insurance Company |
| Please call us at | 1 | Travelers |
| Policy Effective | 1 | Plymouth |
| Policy Number | 1 | Mercury Insurance Company |
| Policy questions or changes. . . . . | 1 | Travelers |
| Policy tier | 1 | Progressive |
| Policy Tier | 1 | Travelers |
| PREMIUMS | 1 | AEIC |
| PRIN/OCCASIONAL | 1 | AEIC |
| PRINCIPAL DRIVER | 1 | NYCM Insurance |
| Private Passenger Automobile | 1 | MAPFRE Insurance Company |
| Protected Since | 1 | NYCM Insurance |
| Protection and Collision premiums for your policy include a | 1 | Foremost Insurance Company |
| PURCHASE DATE | 1 | AEIC |
| questions, additional insurance needs, or claims. | 1 | Travelers |
| QUOTE # | 1 | NYCM Insurance |
| Quoted Premium Amount | 1 | NYCM Insurance |
| Quoted Pro-rated Premium Amount | 1 | NYCM Insurance |
| RATED DRIVER | 1 | NYCM Insurance |
| Recalculated Reconstruction Cost Estimate | 1 | Foremost Insurance Company |
| recovery would then be | 1 | Foremost Insurance Company |
| Recurring Credit Card (RCC) | 1 | Travelers |
| REGISTERED TO | 1 | NYCM Insurance |
| reimbursement up to | 1 | Foremost Insurance Company |
| reimbursement up to . Credit card protection up to | 1 | Foremost Insurance Company |
| REMOVED VEHICLE WITH VIN | 1 | NYCM Insurance |
| REPLACED VEHICLE WITH VIN | 1 | NYCM Insurance |
| REPORT | 1 | NYCM Insurance |
| Required Information Notice, Form | 1 | Mercury Insurance Company |
| Result: Insured recovers | 1 | Foremost Insurance Company |
| Result: Since the other motor vehicle was uninsured, the full | 1 | NYCM Insurance |
| SINGLE LIMIT | 1 | Foremost Insurance Company |
| Spousal Liability | 1 | Foremost Insurance Company |
| SSN | 1 | NYCM Insurance |
| STATUS | 1 | NYCM Insurance |
| Substitute Transportation | 1 | Foremost Insurance Company |
| Subtotal for your vehicle(s): | 1 | Travelers |
| Subtotal policy premium | 1 | Progressive |
| Subtotal Policy Premium (All Vehicles) | 1 | Mercury Insurance Company |
| SUMMARY OF CHANGES | 1 | NYCM Insurance |
| Supplementary Uninsured/Underinsured Motorist | 1 | Mercury Insurance Company |
| SURCHARGE | 1 | NYCM Insurance |
| TERM AMOUNT | 1 | MAPFRE Insurance Company |
| Territory | 1 | NYCM Insurance |
| TERRITORY | 1 | NYCM Insurance |
| the accident, then the insured’s total recovery would be | 1 | NYCM Insurance |
| The additional premium for supplemental spousal liability insurance is | 1 | Foremost Insurance Company |
| The most we will pay for loss to a trailer you do not own is | 1 | Foremost Insurance Company |
| the SUM limits stated in the Declarations or | 1 | Foremost Insurance Company |
| Then the insured's total recovery would be | 1 | Foremost Insurance Company |
| then your total recovery would be | 1 | Travelers |
| This policy was purchased at | 1 | Mercury Insurance Company |
| This results in a maximum payment of | 1 | Foremost Insurance Company |
| TIER | 1 | AEIC |
| Tier | 1 | NYCM Insurance |
| Tier Code | 1 | Mercury Insurance Company |
| Total | 1 | Foremost Insurance Company |
| Total 6 month policy premium if paid in full and fees | 1 | Progressive |
| TOTAL ANNUAL PREMIUM | 1 | Foremost Insurance Company |
| Total Annual SPP Premium | 1 | Foremost Insurance Company |
| Total Discounts | 1 | Mercury Insurance Company |
| Total Policy Premium: (including all discounts and credits) | 1 | Plymouth |
| Total Premium | 1 | Foremost Insurance Company |
| Total Premium for This Policy | 1 | Travelers |
| TOTAL REPLACEMENT SERVICES | 1 | Foremost Insurance Company |
| TOTAL RETURN POLICY PREMIUM | 1 | AEIC |
| Total Savings from these discounts | 1 | Plymouth |
| TOTAL VEHICLE PREMIUM | 1 | NYCM Insurance |
| TOTAL WAGE LOSS UP TO | 1 | Foremost Insurance Company |
| Transaction Expiration | 1 | NYCM Insurance |
| Travelers representative at | 1 | Travelers |
| Type | 1 | Hagerty Insurance |
| TYPE | 1 | NYCM Insurance |
| UMBRELLA POLICY NUMBER | 1 | NYCM Insurance |
| UNDERLYING LIMITS DISCOUNT | 1 | Foremost Insurance Company |
| UNINSURED/UNDERINSURED MOTORISTS | 1 | Foremost Insurance Company |
| UNLICENSED | 1 | NYCM Insurance |
| VEH | 1 | AEIC |
| Vehicle and Traffic Law to: | 1 | Foremost Insurance Company |
| VIOLATIONS | 1 | NYCM Insurance |
| WE WILL CHARGE YOU A | 1 | Foremost Insurance Company |
| will pay you | 1 | Foremost Insurance Company |
| Work Loss - 3 Year Limit | 1 | Foremost Insurance Company |
| work loss, up to | 1 | Foremost Insurance Company |
| would be covered under the SUM Coverage as the total damages | 1 | NYCM Insurance |
| Year | 1 | Plymouth |
| years | 1 | Foremost Insurance Company |
| years (Class D | 1 | Foremost Insurance Company |
| years driving experience | 1 | Plymouth |
| you will have built up a | 1 | Foremost Insurance Company |
| You will receive a | 1 | Foremost Insurance Company |
| Your Account Number: | 1 | Travelers |
| Your annual discount savings is | 1 | Foremost Insurance Company |
| Your current policy will expire on | 1 | Progressive |
| Your Policy Rating Tier is | 1 | Plymouth |
| Your Policy Rating Tier is: | 1 | Plymouth |
| Your Total Savings Reflected in Your Total Premium: | 1 | Travelers |
| • Fax to | 1 | Travelers |

## Notes

- Classic auto (Hagerty) is filed here: guaranteed / agreed value is `vehicles[].stated_amount` ('AGREED VALUE', 'Guaranteed Value'), spare parts is `PA_SPARE_PARTS`.
