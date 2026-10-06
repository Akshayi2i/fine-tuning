# motorcycle - core fields (SPEC_21 4.3)

Core field list of the motorcycle SPEC_21 schema, built from all 9 seed golds of 3 carriers (All State, Progressive, RT Specialty) and the line's L1 registry YAMLs, read together (SPEC_21 4.3, 14.1 Phase 2).

| Item | Value |
|---|---|
| Schema | `motorcycle.json` 1.0.0, composing `common_model.json` 1.0.0 by `$ref` |
| Core leaves | 180 (budget 80-300; `python -m fideon.synth_core.tools leaves`) |
| Printed on the current seeds | 79 of the 180 |
| Seeds | 9 seed golds, 3 carriers: All State, Progressive, RT Specialty |
| Blocks besides the envelope | `locations`, `vehicles`, `drivers`, `rating_modifiers` |
| LOB block | none |
| Coverage codes | 16 (`motorcycle.coverage_codes.yaml`), of them 8 shared `X_` codes |
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
| `document.transaction_type` | Kind of transaction the document records. | - | Progressive, RT Specialty | 5 |
| `document.transaction_date` | Date the transaction was processed. | - | not printed on the current seeds | 0 |
| `document.transaction_effective_date` | Date an endorsement, cancellation or reinstatement takes effect. | - | RT Specialty | 1 |
| `document.transaction_reason` | Reason for the transaction, as printed. | - | not printed on the current seeds | 0 |
| `document.endorsement_number` | Number of the endorsement or policy change. | - | not printed on the current seeds | 0 |
| `document.title` | Document title as printed, e.g. 'Homeowners Policy Declarations'. | - | All State, Progressive, RT Specialty | 9 |
| `document.issue_date` | Date the document was issued. | - | Progressive | 4 |
| `document.print_date` | Date the document was printed. | Printed Date | RT Specialty | 2 |
| `document.mailing_date` | Date the document was mailed; it starts notice periods on cancellation and nonrenewal notices. | - | not printed on the current seeds | 0 |
| `carrier.name` | The writing company. | Underwritten by | Progressive, RT Specialty | 7 |
| `carrier.naic_code` | NAIC company code of the writing company. | - | not printed on the current seeds | 0 |
| `carrier.group_name` | Group or trade name, when printed separately from the writing company. | - | not printed on the current seeds | 0 |
| `carrier.address.street` | Street address line. | - | Progressive, RT Specialty | 5 |
| `carrier.address.street_2` | Second address line: suite, unit, attention line. | - | RT Specialty | 3 |
| `carrier.address.city` | City. | - | Progressive, RT Specialty | 5 |
| `carrier.address.state` | State, two-letter USPS code in parsed. | - | Progressive, RT Specialty | 5 |
| `carrier.address.postal_code` | ZIP code. | - | Progressive, RT Specialty | 5 |
| `carrier.address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.phone` | Main phone number of the carrier. | - | Progressive | 2 |
| `carrier.fax` | Fax number. | - | not printed on the current seeds | 0 |
| `carrier.web_address` | Website address. | - | Progressive | 4 |
| `carrier.claims_phone` | Phone number for reporting a claim. | - | Progressive | 4 |
| `carrier.claims_email` | Address claims are reported to, when printed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street` | Street address line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.city` | City. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.state` | State, two-letter USPS code in parsed. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.postal_code` | ZIP code. | - | not printed on the current seeds | 0 |
| `carrier.claims_address.county` | County. | - | not printed on the current seeds | 0 |
| `carrier.admitted_status` | Whether the writing company is admitted or non-admitted in the state. | - | not printed on the current seeds | 0 |
| `producer.agency_name` | Name of the agency or broker of record. | Agency; Agent Of Record | All State, Progressive, RT Specialty | 7 |
| `producer.producer_code` | Code the carrier assigns to the agency. | AGENCY CODE; Agent ID | All State, RT Specialty | 4 |
| `producer.contact_name` | Contact person at the agency. | - | not printed on the current seeds | 0 |
| `producer.address.street` | Street address line. | - | Progressive, RT Specialty | 5 |
| `producer.address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `producer.address.city` | City. | - | Progressive, RT Specialty | 5 |
| `producer.address.state` | State, two-letter USPS code in parsed. | - | Progressive, RT Specialty | 5 |
| `producer.address.postal_code` | ZIP code. | - | Progressive, RT Specialty | 5 |
| `producer.address.county` | County. | - | not printed on the current seeds | 0 |
| `producer.phone` | Phone number of the agency. | TELEPHONE | Progressive, RT Specialty | 5 |
| `producer.fax` | Fax number. | - | not printed on the current seeds | 0 |
| `producer.email` | Email address. | - | not printed on the current seeds | 0 |
| `producer.web_address` | Website of the agency. | - | not printed on the current seeds | 0 |
| `producer.contract_number` | The agency's contract or sub-code with the carrier. | - | not printed on the current seeds | 0 |
| `named_insured.primary_name` | First named insured. | NAMED INSURED; Named Insured | All State, Progressive, RT Specialty | 9 |
| `named_insured.additional_named_insureds[].name` | Name of the additional named insured. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].relationship` | As printed, e.g. 'Spouse', 'Subsidiary'. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.additional_named_insureds[].occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `named_insured.doing_business_as` | Trade name the insured does business as. | - | not printed on the current seeds | 0 |
| `named_insured.entity_type` | As printed: Individual, Corporation, LLC, Partnership, Trust... | - | not printed on the current seeds | 0 |
| `named_insured.fein` | Federal employer identification number. | - | not printed on the current seeds | 0 |
| `named_insured.business_description` | Nature of the insured's business or operations. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.street` | Street address line. | - | All State, Progressive, RT Specialty | 8 |
| `named_insured.mailing_address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `named_insured.mailing_address.city` | City. | - | All State, Progressive, RT Specialty | 8 |
| `named_insured.mailing_address.state` | State, two-letter USPS code in parsed. | - | All State, Progressive, RT Specialty | 8 |
| `named_insured.mailing_address.postal_code` | ZIP code. | - | All State, Progressive, RT Specialty | 8 |
| `named_insured.mailing_address.county` | County. | - | not printed on the current seeds | 0 |
| `named_insured.phone` | Phone number of the insured. | Preferred Phone | All State | 2 |
| `named_insured.email` | Email address of the insured. | - | not printed on the current seeds | 0 |
| `named_insured.date_of_birth` | Date of birth. | - | not printed on the current seeds | 0 |
| `named_insured.gender` | Gender as printed. | - | not printed on the current seeds | 0 |
| `named_insured.marital_status` | Marital status as printed. | - | not printed on the current seeds | 0 |
| `named_insured.occupation` | Occupation as printed. | - | not printed on the current seeds | 0 |
| `policy.policy_number` | Policy number. | POLICY NUMBER; Policy Number; Policy number | All State, Progressive, RT Specialty | 9 |
| `policy.prior_policy_number` | Number of the policy this one renews or replaces. | - | not printed on the current seeds | 0 |
| `policy.certificate_number` | Certificate number, when the policy is issued as a certificate under a master policy. | - | not printed on the current seeds | 0 |
| `policy.policy_type` | Product, program or policy form name as printed (e.g. 'Homeowners HO-3', 'Auto Special'). | TYPE OF POLICY | All State, Progressive, RT Specialty | 6 |
| `policy.original_inception_date` | Date the insured first became a policyholder with the carrier. | - | Progressive | 1 |
| `policy.effective_date` | Date coverage starts. | POLICY PERIOD; Policy Period; Policy Term | All State, Progressive, RT Specialty | 9 |
| `policy.expiration_date` | Date coverage ends. | - | All State, Progressive, RT Specialty | 8 |
| `policy.term_months` | Term length in months; 'Annual' -> 12. | - | Progressive | 4 |
| `policy.coverage_trigger` | What triggers coverage: occurrence or claims made. | - | not printed on the current seeds | 0 |
| `policy.rating_state` | State the policy is rated in. | RATING STATE | All State, RT Specialty | 3 |
| `policy.audit_period` | As printed: Annual, Semi-Annual, Quarterly, Monthly, Non-Auditable. | - | not printed on the current seeds | 0 |
| `policy.subject_to_audit` | Whether the premium is subject to audit. | - | not printed on the current seeds | 0 |
| `lob_parts[].title` | Title of the coverage part as printed. | - | not printed on the current seeds | 0 |
| `lob_parts[].coverage_part_form` | Coverage-part declarations form number. | - | not printed on the current seeds | 0 |
| `lob_parts[].premium` | Premium for the part. | - | not printed on the current seeds | 0 |
| `coverages[].coverage_name` | As printed. | - | All State, Progressive, RT Specialty | 9 |
| `coverages[].included` | Whether the coverage is included, excluded or not purchased. | - | All State, Progressive, RT Specialty | 9 |
| `coverages[].limits[].amount` | Amount of the limit. | - | All State, Progressive, RT Specialty | 9 |
| `coverages[].limits[].percentage` | When the limit is stated as a percentage of another coverage. | - | not printed on the current seeds | 0 |
| `coverages[].limits[].description` | Printed qualifier, e.g. 'theft of jewelry'. Required in practice for sublimits. | - | All State, Progressive, RT Specialty | 8 |
| `coverages[].deductibles[].amount` | Amount of a flat deductible. | - | All State, Progressive, RT Specialty | 9 |
| `coverages[].deductibles[].percentage` | Percentage deductible, e.g. 2 for a 2% hurricane deductible. | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].percentage_basis` | What the percentage is of, as printed (e.g. 'Coverage A limit'). | - | not printed on the current seeds | 0 |
| `coverages[].deductibles[].peril` | Peril the deductible applies to (wind/hail, hurricane, all other perils). | - | not printed on the current seeds | 0 |
| `coverages[].premium` | Premium for the coverage. | - | All State, Progressive, RT Specialty | 9 |
| `coverages[].valuation` | Valuation basis. | - | Progressive | 2 |
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
| `premium.total` | Total premium for the policy. | TOTAL FULL TERM PREMIUM; Total 12 month policy premium; Policy Premium; TOTAL CHARGES | All State, Progressive, RT Specialty | 9 |
| `premium.deposit` | Deposit premium due at inception. | - | not printed on the current seeds | 0 |
| `premium.minimum` | The least premium the policy (or part) is written for. | MINIMUM WRITTEN | RT Specialty | 1 |
| `premium.minimum_earned` | The least premium kept on cancellation; not the same as `minimum`. | - | RT Specialty | 2 |
| `premium.change` | Premium change made by this transaction; negative for a return premium. | Premium Change | All State | 2 |
| `premium.items[].description` | Description of the premium line as printed. | - | All State, Progressive, RT Specialty | 7 |
| `premium.items[].amount` | Premium amount of the line. | - | All State, Progressive, RT Specialty | 7 |
| `premium.taxes_fees[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.taxes_fees[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.surcharges[].description` | Name of the tax, fee, surcharge or discount as printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].amount` | Amount of the tax, fee, surcharge or discount. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.surcharges[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `premium.discounts[].description` | Name of the tax, fee, surcharge or discount as printed. | - | Progressive | 2 |
| `premium.discounts[].amount` | Amount of the tax, fee, surcharge or discount. | Discount if paid in full | Progressive | 2 |
| `premium.discounts[].percentage` | Rate of the charge or discount as a percentage, when printed. | - | not printed on the current seeds | 0 |
| `premium.discounts[].fully_earned` | Whether the charge is fully earned (not refunded on cancellation). | - | not printed on the current seeds | 0 |
| `billing.payment_plan` | Payment plan as printed (annual, 2-pay, 12-pay EFT...). | Pay Plan | All State, Progressive | 3 |
| `billing.bill_type` | As printed: Direct Bill, Agency Bill. | - | not printed on the current seeds | 0 |
| `billing.payment_method` | As printed: EFT, recurring card, check... | Pay Method | All State | 2 |
| `billing.account_number` | Billing account number the carrier bills under. | - | not printed on the current seeds | 0 |
| `billing.amount_due` | Amount currently due on the bill. | Current Amount Due | All State, Progressive | 3 |
| `billing.due_date` | Date payment is due. | Due Date; Renewal Payment Due By | All State, Progressive | 3 |
| `billing.installments[].installment_number` | Number of the installment in the payment plan. | - | not printed on the current seeds | 0 |
| `billing.installments[].due_date` | Date the installment is due. | - | Progressive | 1 |
| `billing.installments[].amount` | Amount of the installment. | - | Progressive | 1 |
| `forms_and_endorsements[].form_number` | Form number as printed, e.g. 'HO 00 03'. | - | Progressive, RT Specialty | 6 |
| `forms_and_endorsements[].edition_date` | Edition date of the form as printed (often month and year only, e.g. '05 11'). | - | Progressive | 4 |
| `forms_and_endorsements[].title` | Title of the form or endorsement. | - | not printed on the current seeds | 0 |
| `forms_and_endorsements[].premium` | Premium charged for the form or endorsement. | - | not printed on the current seeds | 0 |
| `locations[].location_number` | Location number as printed on the schedule. | - | not printed on the current seeds | 0 |
| `locations[].address.street` | Street address line. | - | All State, RT Specialty | 4 |
| `locations[].address.street_2` | Second address line: suite, unit, attention line. | - | not printed on the current seeds | 0 |
| `locations[].address.city` | City. | - | All State, RT Specialty | 4 |
| `locations[].address.state` | State, two-letter USPS code in parsed. | - | All State, RT Specialty | 4 |
| `locations[].address.postal_code` | ZIP code. | Garaging Zip Code | All State, Progressive, RT Specialty | 7 |
| `locations[].address.county` | County. | - | not printed on the current seeds | 0 |
| `locations[].territory` | Rating territory or zone. | - | RT Specialty | 1 |
| `locations[].protection_class` | Fire protection class of the premises. | - | not printed on the current seeds | 0 |
| `locations[].fire_district` | Fire district or fire protection area the premises fall in. | - | not printed on the current seeds | 0 |
| `locations[].insured_interest` | The insured's interest in the premises: Owner, Deeded owner, Tenant, LLC member... | - | not printed on the current seeds | 0 |
| `locations[].distance_to_fire_station` | Distance to the responding fire station. | - | not printed on the current seeds | 0 |
| `locations[].distance_to_hydrant` | Distance to the nearest fire hydrant. | - | not printed on the current seeds | 0 |
| `vehicles[].vehicle_number` | Vehicle number as printed on the schedule. | - | RT Specialty | 3 |
| `vehicles[].year` | Model year. | - | All State, Progressive, RT Specialty | 9 |
| `vehicles[].make` | Manufacturer. | - | All State, Progressive, RT Specialty | 9 |
| `vehicles[].model` | Model. | - | All State, Progressive, RT Specialty | 9 |
| `vehicles[].vin` | Vehicle identification number. | VIN | All State, Progressive, RT Specialty | 9 |
| `vehicles[].body_type` | A trailer is a vehicle with body_type 'Trailer'. | - | not printed on the current seeds | 0 |
| `vehicles[].use` | How the vehicle is used (pleasure, commute, business...). | Usage Class | All State | 2 |
| `vehicles[].annual_mileage` | As printed; often a band ('10,000 - 11,999'). | Annual Miles | All State | 2 |
| `vehicles[].commute_miles_one_way` | One-way commuting distance printed for the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].engine_displacement` | Engine size as printed (e.g. '1,745 cc'). | - | Progressive, RT Specialty | 7 |
| `vehicles[].cost_new` | Original cost new of the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].stated_amount` | Stated or agreed value of the vehicle. | - | not printed on the current seeds | 0 |
| `vehicles[].symbols[]` | Rating symbols printed for the vehicle. | - | not printed on the current seeds | 0 |
| `drivers[].driver_number` | Driver number as printed on the schedule. | - | RT Specialty | 2 |
| `drivers[].name` | Driver's name. | Named Insured | All State, Progressive, RT Specialty | 9 |
| `drivers[].date_of_birth` | Date of birth. | - | RT Specialty | 3 |
| `drivers[].age` | Driver's age as printed. | - | Progressive | 4 |
| `drivers[].gender` | Gender as printed. | - | All State, Progressive | 6 |
| `drivers[].marital_status` | Marital status as printed. | - | All State, Progressive | 6 |
| `drivers[].license_state` | State that issued the driver's licence. | - | not printed on the current seeds | 0 |
| `drivers[].license_number` | Driver's licence number. | - | not printed on the current seeds | 0 |
| `drivers[].relationship` | Relationship to the named insured. | - | All State, Progressive | 6 |
| `drivers[].status` | Driver status (rated, excluded, listed). | - | All State | 2 |
| `rating_modifiers[].description` | As printed, e.g. 'Experience Modification', 'Multi-Policy Discount'. | - | All State, Progressive, RT Specialty | 8 |
| `rating_modifiers[].factor` | Modification factor (e.g. 0.85). | - | not printed on the current seeds | 0 |
| `rating_modifiers[].percent` | Modifier as a percentage. | - | not printed on the current seeds | 0 |

## Printed labels with no core field yet (overflow)

Labels the seed golds keep in `additional_fields` (SPEC_21 4.5), with the number of carriers that print
them. A label printed by two or more carriers is a candidate for a core field in a later minor version (0 today).
Labels that would carry a printed name, address or number are left out of this list. Some engine-test golds
took the text printed before a value as its label, so the list also holds prose fragments and schedule
descriptions; those labels are corrected at seed review, not promoted.

| Label | Carriers | Carrier names |
|---|---|---|
| AAP | 1 | All State |
| Accessories | 1 | Progressive |
| ACCESSORIES & NON-STANDARD EQUIPMENT SCHEDULE | 1 | RT Specialty |
| Accident Waiver Enhancement | 1 | All State |
| Address Standardization | 1 | All State |
| Agent Of Record | 1 | All State |
| Allstate Easy Pay Plan Enrollment | 1 | All State |
| Allstate eBill | 1 | All State |
| Allstate ePolicy | 1 | All State |
| American Reliable Insurance Company | 1 | RT Specialty |
| An installment fee of | 1 | Progressive |
| Anti-Lock Brakes | 1 | Progressive |
| Application Date | 1 | All State |
| Application Of Origin | 1 | All State |
| Application Time | 1 | All State |
| Balance | 1 | All State |
| Billing Group | 1 | All State |
| Bind Date | 1 | All State |
| Bind ID | 1 | All State |
| Bind Time | 1 | All State |
| Buyout Indicator | 1 | All State |
| Channel Of Bind | 1 | All State |
| Channel Of Origin | 1 | All State |
| Channel Of Process | 1 | All State |
| Claim Free Renewal | 1 | Progressive |
| Company-Line | 1 | All State |
| Control Number | 1 | All State |
| Current Address | 1 | All State |
| Customer service and claims service | 1 | Progressive |
| Date Moved To Current Address | 1 | All State |
| Date of Birth | 1 | All State |
| Discounted AAP | 1 | All State |
| Front End Matched | 1 | All State |
| Garaging Zip Code | 1 | Progressive |
| Geocode Current Address | 1 | All State |
| Home Owner | 1 | Progressive |
| Includes savings of | 1 | Progressive |
| Last Bill Amount | 1 | All State |
| LIMIT/DEDUCTIBLE | 1 | RT Specialty |
| Multi-Policy | 1 | Progressive |
| Multi-Vehicle | 1 | Progressive |
| Named Insured | 1 | All State |
| New Motorcycle Expanded Protection | 1 | All State |
| Number Of Times Renewed | 1 | All State |
| OPT BASIC ECON LOSS | 1 | RT Specialty |
| Original Year | 1 | All State |
| Ownership Type | 1 | All State |
| Page footer date (no printed label) | 1 | RT Specialty |
| Paid in Full | 1 | Progressive |
| PAY IN FULL | 1 | Progressive |
| PAY IN INSTALLMENTS | 1 | Progressive |
| Policy | 1 | RT Specialty |
| Policy Rate Control | 1 | All State |
| Policy tier | 1 | Progressive |
| Premium | 1 | All State |
| Premium At Last Renewal | 1 | All State |
| Premium At Last Renewal Date | 1 | All State |
| Premium At Renewal | 1 | All State |
| Premium At Renewal Date | 1 | All State |
| Premium Change Percent | 1 | All State |
| Primary Residence | 1 | All State |
| Printed Date | 1 | RT Specialty |
| Producer Name | 1 | All State |
| Required Information Notice, Form | 1 | Progressive |
| rounded to the nearest | 1 | RT Specialty |
| Safe Driving Deductible Reward | 1 | All State |
| SAFETY CLOTHING & TOWING | 1 | RT Specialty |
| SEE ENCLOSURE FOR | 1 | Progressive |
| Source Of Quote | 1 | All State |
| SR Tier | 1 | All State |
| STATUS | 1 | All State |
| SUMMARY OF COVERAGES | 1 | RT Specialty |
| Surcharges Applied | 1 | All State |
| Terr. | 1 | RT Specialty |
| This policy period ends on | 1 | Progressive |
| Tier/Group | 1 | All State |
| Total 12 month policy premium if paid in full | 1 | Progressive |
| TOTAL FULL TERM PREMIUM | 1 | RT Specialty |
| Total of Accessories (rounded to the nearest $100) | 1 | RT Specialty |
| Trailer | 1 | RT Specialty |
| Transfer | 1 | Progressive |
| Unit | 1 | RT Specialty |
| Usage | 1 | All State |
| Version Number | 1 | All State |
| Your Choice Package | 1 | All State |
| Your coverage began on | 1 | Progressive |
| Your current policy will expire on | 1 | Progressive |
| your Policy and Endorsement forms available at | 1 | RT Specialty |
