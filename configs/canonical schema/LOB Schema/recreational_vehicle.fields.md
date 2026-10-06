# recreational_vehicle - core fields (SPEC_21 4.3)

Core field list of the recreational_vehicle SPEC_21 schema, built from all 22 seed golds of 4 carriers (All State, American Modern, Foremost Insurance Company, Progressive) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `recreational_vehicle.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 180 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 95 of the 180 |
| Seeds | 22 seed golds, 4 carriers: All State, American Modern, Foremost Insurance Company, Progressive |
| Blocks besides the envelope | `locations`, `vehicles`, `drivers`, `rating_modifiers` |
| LOB block | none |
| Coverage codes | 26 (`recreational_vehicle.coverage_codes.yaml`), of them 8 shared `X_` codes |
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
| `document.transaction_type` | Kind of transaction the document records. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `document.transaction_date` | Date the transaction was processed. | TRANSACTION DATE | All State | 1 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | TRANSACTION EFFECTIVE DATE; Transaction Effective Date | All State, American Modern, Foremost Insurance Company | 3 |
| `document.transaction_reason` | Reason for the transaction, as printed. | - | not printed on the current seeds | 0 |
| `document.endorsement_number` | Number of the endorsement or policy change. | - | not printed on the current seeds | 0 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `document.issue_date` | Date the document was issued. | - | All State, Foremost Insurance Company, Progressive | 11 |
| `document.print_date` | Date the document was printed. | - | All State, American Modern | 6 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | not printed on the current seeds | 0 |
| `carrier.name` | The writing company. | Underwritten by | All State, American Modern, Foremost Insurance Company, Progressive | 17 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | American Modern, Foremost Insurance Company, Progressive | 12 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.address.city` | City. | - | American Modern, Foremost Insurance Company, Progressive | 12 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | American Modern, Foremost Insurance Company, Progressive | 12 |
| `carrier.address.postal_code` | ZIP code. | - | American Modern, Foremost Insurance Company, Progressive | 12 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | Progressive | 6 |
| `carrier.fax` | Fax number. | - | not printed on the current seeds | 0 |
| `carrier.web_address` | Website address. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `carrier.claims_phone` | Phone number for reporting a claim. | FOR CLAIMS CALL; Report a Claim | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | Agency; Agent Of Record; Your Agent | All State, American Modern, Foremost Insurance Company, Progressive | 21 |
| `producer.producer_code` | Code the carrier assigns to the agency. | AGENCY ID; Agency Code; Agent ID | All State, American Modern, Foremost Insurance Company | 12 |
| `producer.contact_name` | Contact person at the agency. | - | not printed on the current seeds | 0 |
| `producer.address.street` | Street address line. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `producer.address.city` | City. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `producer.address.postal_code` | ZIP code. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | Telephone | All State, American Modern, Foremost Insurance Company, Progressive | 15 |
| `producer.fax` | Fax number. | - | not printed on the current seeds | 0 |
| `producer.email` | Email address. | - | not printed on the current seeds | 0 |
| `producer.web_address` | Website of the agency. | - | not printed on the current seeds | 0 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | Named Insured; Named Insured(s); NAMED INSURED(S); Insured's Name; PolicyHolder | All State, American Modern, Progressive | 21 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | All State, Progressive | 13 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.city` | City. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | Preferred Phone; Cell Phone; Business Phone | All State | 6 |
| `named_insured.email` | Email address of the insured. | Email Address | All State | 2 |
| `named_insured.date_of_birth` | Date of birth. | Date of Birth | All State | 1 |
| `named_insured.gender` | Gender as printed. | Gender; Sex | All State | 1 |
| `named_insured.marital_status` | Marital status as printed. | Marital Status | All State | 1 |
| `named_insured.occupation` | Occupation as printed. | Occupation | All State | 1 |
| `policy.policy_number` | Policy number. | Policy Number; Policy number; POLICY NUMBER; PolicyNumber | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | POLICY TYPE; Policy Type | All State, American Modern, Foremost Insurance Company, Progressive | 20 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | - | All State, Progressive | 6 |
| `policy.effective_date` | Date coverage starts. | Policy Period; EFFECTIVE DATE; EFF. DATE; INCEPTION DATE | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `policy.expiration_date` | Date coverage ends. | EXPIRATION DATE | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | - | All State, Progressive | 10 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | RATING STATE | All State | 5 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | not printed on the current seeds | 0 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | not printed on the current seeds | 0 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | All State, American Modern, Progressive | 16 |
| `coverages[].limits[].amount` | Amount of the limit. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | - | All State, American Modern, Foremost Insurance Company, Progressive | 18 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | Progressive | 3 |
| `coverages[].premium` | Premium for the coverage. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `coverages[].valuation` | Valuation basis. | - | All State, American Modern, Progressive | 11 |
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
| `premium.total` | Total premium for the policy. | Total 12 month policy premium; Total Policy Premium; Policy Premium; TERM AMOUNT | All State, American Modern, Progressive | 19 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | - | not printed on the current seeds | 0 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | MINIMUM EARNED PREMIUM | American Modern, Foremost Insurance Company | 6 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Premium Change | All State | 5 |
| `premium.items[].description` | Description of the premium line as printed. | - | All State, American Modern, Progressive | 17 |
| `premium.items[].amount` | Premium amount of the line. | - | All State, American Modern, Progressive | 17 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | American Modern, Foremost Insurance Company, Progressive | 6 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | LAW ENFORCEMENT FEE; Motor vehicle law enforcement fee | American Modern, Foremost Insurance Company, Progressive | 6 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | All State, American Modern, Progressive | 13 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | Discount if paid in full; Multi-Vehicle Discount | All State, American Modern, Progressive | 12 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | Pay Plan; Payment Plan | All State, Foremost Insurance Company, Progressive | 13 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | not printed on the current seeds | 0 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | Pay Method | All State | 5 |
| `billing.account_number` | Billing account number the carrier bills under. | - | not printed on the current seeds | 0 |
| `billing.amount_due` | Amount currently due on the bill. | Current Amount Due | All State, Progressive | 11 |
| `billing.due_date` | Date payment is due. | Due Date; Renewal Payment Due By | All State, Progressive | 11 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | Date Due | Foremost Insurance Company, Progressive | 7 |
| `billing.installments[].amount` | Amount of the installment. | - | Foremost Insurance Company, Progressive | 7 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | All State, American Modern, Foremost Insurance Company, Progressive | 16 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | All State, Progressive | 10 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | All State, American Modern, Foremost Insurance Company, Progressive | 14 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | not printed on the current seeds | 0 |
| `locations[].location_number` | Location number as printed on the schedule. | - | All State | 1 |
| `locations[].address.street` | Street address line. | - | All State, Foremost Insurance Company | 7 |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `locations[].address.city` | City. | - | All State, Foremost Insurance Company | 7 |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | All State, Foremost Insurance Company | 7 |
| `locations[].address.postal_code` | ZIP code. | Garaging Zip Code | All State, Foremost Insurance Company, Progressive | 17 |
| `locations[].address.county` | County. | - | Foremost Insurance Company | 1 |
| `locations[].territory` | Rating territory or zone. | Territory | All State, Foremost Insurance Company | 2 |
| `locations[].protection_class` | Fire protection class of the premises. | - | not printed on the current seeds | 0 |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | - | not printed on the current seeds | 0 |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | not printed on the current seeds | 0 |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | - | not printed on the current seeds | 0 |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | - | not printed on the current seeds | 0 |
| `vehicles[].vehicle_number` | Vehicle number as printed on the schedule. | Vehicle #; UNIT # | American Modern, Foremost Insurance Company | 6 |
| `vehicles[].year` | Model year. | - | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `vehicles[].make` | Manufacturer. | - | All State, Foremost Insurance Company, Progressive | 16 |
| `vehicles[].model` | Model. | YEAR/MAKE/MODEL | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `vehicles[].vin` | Vehicle identification number. | VIN; Vehicle ID Number | All State, American Modern, Foremost Insurance Company, Progressive | 22 |
| `vehicles[].body_type` | A trailer is a vehicle with body_type 'Trailer'. | - | All State, Progressive | 4 |
| `vehicles[].use` | How the vehicle is used (pleasure, commute, business...). | Usage Class | All State | 5 |
| `vehicles[].annual_mileage` | As printed; often a band ('10,000 - 11,999'). | Annual Miles | All State | 5 |
| `vehicles[].commute_miles_one_way` | One-way commuting distance printed for the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].engine_displacement` | Engine size as printed (e.g. '1,745 cc'). | - | All State, American Modern, Progressive | 12 |
| `vehicles[].cost_new` | Original cost new of the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].stated_amount` | Stated or agreed value of the vehicle. | - | Foremost Insurance Company | 1 |
| `vehicles[].symbols[]` | Rating symbols printed for the vehicle. | Price Group; Price Group Symbol | All State | 2 |
| `drivers[].driver_number` | Driver number as printed on the schedule. | - | All State, Foremost Insurance Company | 2 |
| `drivers[].name` | Driver's name. | Listed Drivers | All State, American Modern, Progressive | 21 |
| `drivers[].date_of_birth` | Date of birth. | Date of Birth | All State | 1 |
| `drivers[].age` | Driver's age as printed. | - | All State, Progressive | 10 |
| `drivers[].gender` | Gender as printed. | Gender; Sex | All State, Progressive | 16 |
| `drivers[].marital_status` | Marital status as printed. | Marital Status | All State, Foremost Insurance Company, Progressive | 17 |
| `drivers[].license_state` | State that issued the driver's licence. | License State / Number | All State, Foremost Insurance Company | 2 |
| `drivers[].license_number` | Driver's licence number. | - | All State, Foremost Insurance Company | 2 |
| `drivers[].relationship` | Relationship to the named insured. | Relationship | All State, Progressive | 16 |
| `drivers[].status` | Driver status (rated, excluded, listed). | - | All State | 5 |
| `rating_modifiers[].description` | As printed, e.g. 'Experience Modification', 'Multi-Policy Discount'. | - | All State, Progressive | 12 |
| `rating_modifiers[].factor` | Modification factor (e.g. 0.85). | - | not printed on the current seeds | 0 |
| `rating_modifiers[].percent` | Modifier as a percentage. | - | not printed on the current seeds | 0 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (5 today: 1st Offense; CTR within 5 years of a; Insured's bodily injury damage claim paid to spouse; Maximum; Minimum).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| 1st Offense | 2 | American Modern, Foremost Insurance Company |
| CTR within 5 years of a | 2 | American Modern, Foremost Insurance Company |
| Insured's bodily injury damage claim paid to spouse | 2 | American Modern, Foremost Insurance Company |
| Maximum | 2 | American Modern, Foremost Insurance Company |
| Minimum | 2 | American Modern, Foremost Insurance Company |
| (no printed label) | 1 | American Modern |
| (no printed label: page-2 footer time stamp) | 1 | All State |
| 1-PAY 2-PAY 4-PAY 10-PAY 12-PAY | 1 | Foremost Insurance Company |
| 1. Up to | 1 | American Modern |
| 10 Years (Class E Maximum | 1 | American Modern |
| 100 fee to terminate suspension | 1 | American Modern |
| 1st Offense civil penalty and N/A 6-Month Suspension | 1 | Foremost Insurance Company |
| 2. 80% of lost earnings up to a maximum monthly payment of | 1 | American Modern |
| 3. up to | 1 | American Modern |
| 3rd Offense or more Minimum | 1 | Foremost Insurance Company |
| 4. Up to | 1 | American Modern |
| 888 number provided | 1 | American Modern |
| = $275,000) are less than the $300,000 CSL SUM limit. | 1 | American Modern |
| AAA Driver Improvement Program | 1 | Foremost Insurance Company |
| AAP | 1 | All State |
| AARP | 1 | Foremost Insurance Company |
| accident for injured persons and per person and | 1 | American Modern |
| Accident Prevention Course | 1 | Progressive |
| Address Standardization | 1 | All State |
| Agent Of Record | 1 | All State |
| Agents Office | 1 | American Modern |
| Allstate eBill | 1 | All State |
| Allstate ePolicy | 1 | All State |
| AMERICAN AUTOMOBILE ASSOCIATION (AAA) | 1 | Foremost Insurance Company |
| AMERICAN SAFETY | 1 | Foremost Insurance Company |
| AMERICAN SAFETY COUNCIL | 1 | Foremost Insurance Company |
| AMERICAN SAFETY COUNCIL (Available in English and Spanish) | 1 | Foremost Insurance Company |
| Amount Enclosed | 1 | Foremost Insurance Company |
| An installment fee of | 1 | Progressive |
| AN ONLINE DEFENSIVE DRIVING COURSE BY IMPROV | 1 | Foremost Insurance Company |
| and SUM limits of $150,000 or more, the SUM recovery would then be | 1 | American Modern |
| and, subject to this per person limit | 1 | American Modern |
| Another Passenger's Damages that resulted in death | 1 | American Modern |
| Anti-Lock Brakes | 1 | Progressive |
| Application Date | 1 | All State |
| Application Of Bind | 1 | All State |
| Application Of Origin | 1 | All State |
| Application Of Process | 1 | All State |
| Application Time | 1 | All State |
| Automated Customer Service and Direct Customer Care | 1 | American Modern |
| Balance | 1 | All State |
| Billing Group | 1 | All State |
| Bind Date | 1 | All State |
| Bind ID | 1 | All State |
| Bind Time | 1 | All State |
| Birth Date | 1 | Foremost Insurance Company |
| Business Phone | 1 | All State |
| Buyout Indicator | 1 | All State |
| CALL | 1 | American Modern |
| certain expenses, up to | 1 | American Modern |
| Channel Of Bind | 1 | All State |
| Channel Of Origin | 1 | All State |
| Channel Of Process | 1 | All State |
| charged a | 1 | Foremost Insurance Company |
| Chemical Test Refusal | 1 | American Modern |
| Claim Free Renewal | 1 | Progressive |
| Class Code | 1 | All State |
| Company-Line | 1 | All State |
| contact your agent or call our executive office at | 1 | American Modern |
| Control Number | 1 | All State |
| coverage as follows | 1 | American Modern |
| CTR - under Zero | 1 | Foremost Insurance Company |
| CTR-under Zero | 1 | American Modern |
| Current Address | 1 | All State |
| Current Amount Due Includes | 1 | Foremost Insurance Company |
| damages amounted to | 1 | American Modern |
| Date Moved To Current Address | 1 | All State |
| Date of Birth | 1 | All State |
| Daytime Running Lamps | 1 | Progressive |
| DDP enrollment, and the DDP charges an additional course fee of up to | 1 | American Modern |
| Dear | 1 | Foremost Insurance Company |
| Defensive Driver | 1 | All State |
| Discounted AAP | 1 | All State |
| Distant Student | 1 | All State |
| Driver | 1 | All State |
| Driver Training | 1 | All State |
| DRIVER TRAINING ASSOCIATES | 1 | Foremost Insurance Company |
| DRIVER TRAINING ASSOCIATES (DTA) | 1 | Foremost Insurance Company |
| driver’s license by DMV. You also will be charged a | 1 | American Modern |
| EMPIRE SAFETY COUNCIL | 1 | Foremost Insurance Company |
| enrollment, and the DDP charges an additional course fee of up to | 1 | Foremost Insurance Company |
| fine of | 1 | American Modern |
| fine of to | 1 | American Modern |
| For billing questions call our automated phone service, at | 1 | Foremost Insurance Company |
| from the negligent owner or operator of the other motor vehicle and | 1 | Foremost Insurance Company |
| Front End Matched | 1 | All State |
| FS-20 | 1 | Foremost Insurance Company |
| Geocode Current Address | 1 | All State |
| Good Driver | 1 | All State |
| greater of the SUM limits stated in the Declarations or | 1 | American Modern |
| had no liability insurance at all, the insured would collect | 1 | American Modern |
| However, we will not pay more than | 1 | American Modern |
| I DRIVE SAFELY | 1 | Foremost Insurance Company |
| If the insured's CSL and CSL SUM limit were each | 1 | American Modern |
| Includes savings of | 1 | Progressive |
| infraction as opposed to a crime) must pay a | 1 | Foremost Insurance Company |
| insurance agent or broker, or call our toll-free telephone number | 1 | American Modern |
| Insured's Bodily Injury Damages | 1 | American Modern |
| Insured's Bodily Injury Damages ........ | 1 | Foremost Insurance Company |
| Insured's bodily injury Policy coverage limit | 1 | American Modern |
| Insured's bodily injury policy coverage limit | 1 | American Modern |
| Insured's Combined Single Liability (CSL) Limit | 1 | American Modern |
| Insured's Liability Limit | 1 | American Modern |
| Insured's Liability Limit ....................... | 1 | Foremost Insurance Company |
| Insured's SUM Limit | 1 | American Modern |
| Insured's SUM Limit ............................. | 1 | Foremost Insurance Company |
| Last Bill Amount | 1 | All State |
| Late Charge of | 1 | American Modern |
| Length | 1 | Foremost Insurance Company |
| LOCATION 1 | 1 | All State |
| MAIL THIS CARD WITH YOUR PAYMENT TO | 1 | Foremost Insurance Company |
| maximum of per person | 1 | American Modern |
| Merit Points | 1 | All State |
| months for a CDL) and must pay a civil penalty of | 1 | American Modern |
| most we will pay in any one occurrence is | 1 | American Modern |
| motor vehicle and | 1 | American Modern |
| Multi-Policy | 1 | Progressive |
| Multi-Snowmobile | 1 | Progressive |
| must pay a | 1 | American Modern |
| Named Insured | 1 | All State |
| NATIONAL POINT AND INSURANCE REDUCTION COURSE | 1 | Foremost Insurance Company |
| NATIONAL SAFETY COUNCIL | 1 | Foremost Insurance Company |
| NATIONAL TRAFFIC SAFETY INSTITUTE | 1 | Foremost Insurance Company |
| NATIONAL TRAFFIC SAFETY INSTITUTE (NTSI) | 1 | Foremost Insurance Company |
| NEW YORK DRIVER, INC. | 1 | Foremost Insurance Company |
| NEW YORK SAFETY PROGRAM | 1 | Foremost Insurance Company |
| NEW YORK SAFETY PROGRAM (NYSP) | 1 | Foremost Insurance Company |
| NEW YORK STATE INSURANCE IDENTIFICATION CARD | 1 | Foremost Insurance Company |
| Non Sufficient Funds (NSF) Charge of | 1 | American Modern |
| Number Of Times Renewed | 1 | All State |
| NYDDC LLC | 1 | Foremost Insurance Company |
| Odometer | 1 | All State |
| of | 1 | American Modern |
| or collision coverage, we provide up to | 1 | Progressive |
| OR, TO PAY IN FULL, PAY | 1 | Foremost Insurance Company |
| Original Year | 1 | All State |
| other claimants subject to a maximum of per person | 1 | Foremost Insurance Company |
| Other Motor Vehicle Liability Limit | 1 | American Modern |
| Other Motor Vehicle Liability Limit ..... | 1 | Foremost Insurance Company |
| Ownership Type | 1 | All State |
| Paid in Full | 1 | Progressive |
| Part, we will pay up to | 1 | American Modern |
| pay for transporting in any one occurrence is | 1 | American Modern |
| PAY IN FULL | 1 | Progressive |
| PAY IN INSTALLMENTS | 1 | Progressive |
| Pay initial installment: | 1 | Progressive |
| payment. As always, simply call our billing service at | 1 | Foremost Insurance Company |
| per person | 1 | Foremost Insurance Company |
| PHONE | 1 | American Modern |
| phone \| 8 a.m. to 8 p.m. Eastern | 1 | American Modern |
| Plus, your rate went down by | 1 | Progressive |
| POLICY NUMBER EFFECTIVE DATE EXPIRATION DATE | 1 | Foremost Insurance Company |
| Policy Premium | 1 | All State |
| Policy Rate Control | 1 | All State |
| Policy tier | 1 | Progressive |
| Premium | 1 | All State |
| Premium At Last Renewal | 1 | All State |
| Premium At Last Renewal Date | 1 | All State |
| Premium At Renewal | 1 | All State |
| Premium At Renewal Date | 1 | All State |
| Premium Change Percent | 1 | All State |
| Premium surcharges | 1 | All State |
| Primary Residence | 1 | All State |
| Producer Name | 1 | All State |
| Prompt Payment | 1 | Progressive |
| Purchase | 1 | Foremost Insurance Company |
| Questions? Call | 1 | American Modern |
| Rated Driver | 1 | All State |
| Recreational Vehicles | 1 | All State |
| Reinstatement Charge of | 1 | American Modern |
| REPRESENTATIVE NO. | 1 | Foremost Insurance Company |
| Requested Agent | 1 | All State |
| required fees, and meet other eligibility requirements. DMV charges a | 1 | American Modern |
| Result: Insured recovers | 1 | Foremost Insurance Company |
| Safety Course | 1 | Progressive |
| Service Charge of | 1 | American Modern |
| Source Of Quote | 1 | All State |
| SR Tier | 1 | All State |
| STATUS | 1 | All State |
| Statutory Uninsured Motorists Coverage can pay up to | 1 | American Modern |
| SUM coverage is available in the following limits | 1 | Foremost Insurance Company |
| Surcharges Applied | 1 | All State |
| TERM AMOUNT | 1 | All State |
| Territory | 1 | All State |
| The additional | 1 | American Modern |
| then be | 1 | Foremost Insurance Company |
| then the insured's total recovery would be | 1 | Foremost Insurance Company |
| Tier/Group | 1 | All State |
| Tolerance Law (ZTL) | 1 | American Modern |
| Total 12 month policy premium and fees | 1 | Progressive |
| Total 12 month policy premium if paid in full | 1 | Progressive |
| Total 12 month policy premium if paid in full and fees | 1 | Progressive |
| Total general policy coverage | 1 | Progressive |
| TOTAL PREMIUM AND OTHER AMOUNTS FOR THIS POLICY PERIOD | 1 | Foremost Insurance Company |
| TRANS TYPE | 1 | Foremost Insurance Company |
| Transfer | 1 | Progressive |
| UNIT #1 TOTAL PREMIUM AND OTHER AMOUNTS | 1 | Foremost Insurance Company |
| USA TRAINING COMPANY | 1 | Foremost Insurance Company |
| Usage | 1 | All State |
| Version Number | 1 | All State |
| we will pay in any one occurrence is | 1 | American Modern |
| We will pay up to | 1 | American Modern |
| with a | 1 | American Modern |
| Your current policy will expire on | 1 | Progressive |
