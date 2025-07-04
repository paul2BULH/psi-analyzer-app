import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set

class PSICalculator:
    """
    A class to calculate Patient Safety Indicators (PSIs) based on provided
    patient encounter data and a comprehensive set of appendix codes.

    This framework handles:
    1. Loading all required code reference sets from a specified appendix file.
    2. Parsing and validating patient encounter data.
    3. Applying base exclusion logic common to many PSIs.
    4. Returning structured results for each PSI evaluation.
    """

    def __init__(self, codes_source_path: str = 'PSI_Code_Sets.json', psi_definitions_path: str = 'PSI_02_19_Compiled_Cleaned.json'):
        """
        Initializes the PSICalculator with code sets and PSI definitions.

        Args:
            codes_source_path (str): Path to the JSON file containing code sets.
            psi_definitions_path (str): Path to the JSON file containing PSI definitions.
        """
        self.code_sets = self._load_code_sets(codes_source_path)
        self.psi_definitions = self._load_psi_definitions(psi_definitions_path)

        # --- FIX: Updated diagnosis and POA column names as per the prompt's input data structure ---
        # Pdx is the principal diagnosis, DX1-DX25 are secondary diagnoses
        self.dx_cols = ['Pdx'] + [f"DX{i}" for i in range(1, 26)] # Pdx, DX1 to DX25
        # POA1 corresponds to Pdx, POA2 to DX1, ..., POA25 to DX24. DX25 will not have a POA column.
        self.poa_cols = [f"POA{i}" for i in range(1, 26)] # POA1 to POA25

        self.proc_cols = [f"Proc{i}" for i in range(1, 11)] # Proc1 to Proc10
        self.proc_date_cols = [f"Proc{i}_Date" for i in range(1, 11)] # Proc1_Date to Proc10_Date
        self.proc_time_cols = [f"Proc{i}_Time" for i in range(1, 11)] # Proc1_Time to Proc10_Time

        # PSI_03 Specific Anatomic Site Mappings
        self.anatomic_site_map: Dict[str, str] = {
            'PIRELBOWD': 'DTIRELBOEXD',
            'PILELBOWD': 'DTILELBOEXD',
            'PIRUPBACKD': 'DTIRUPBACEXD',
            'PILUPBACKD': 'DTILUPBACEXD',
            'PIRLOBACKD': 'DTIRLOBACEXD',
            'PILLOBACKD': 'DTILLOBACEXD',
            'PISACRALD': 'DTISACRAEXD',
            'PIRHIPD': 'DTIRHIPEXD',
            'PILHIPD': 'DTILHIPEXD',
            'PIRBUTTD': 'DTIRBUTEXD',
            'PILBUTTD': 'DTILBUTEXD',
            'PICONTIGBBHD': 'DTICONTBBHEXD',
            'PIRANKLED': 'DTIRANKLEXD',
            'PILANKLED': 'DTILANKLEXD',
            'PIRHEELD': 'DTIRHEELEXD',
            'PILHEELD': 'DTILHEELEXD',
            'PIHEADD': 'DTIHEADEXD',
            'PIOTHERD': 'DTIOTHEREXD',
        }
        self.unspecified_pu_codes: Set[str] = {
            'PINELBOWD', 'PINBACKD', 'PINHIPD', 'PINBUTTD',
            'PINANKLED', 'PINHEELD', 'PIUNSPECD'
        }
        # Union of all specific pressure ulcer codes
        self.all_specific_pu_codes: Set[str] = set(self.anatomic_site_map.keys())
        # Union of all DTI exclusion codes
        self.all_dti_ex_codes: Set[str] = set(self.anatomic_site_map.values())
        # PI~EXD* codes for principal/POA=Y secondary exclusion (union of PI and DTI exclusions)
        # Based on the JSON, PI~EXD* refers to the general pressure ulcer/DTI exclusion codes
        self.pi_exd_codes_for_principal_exclusion: Set[str] = set()
        for header in self.code_sets.keys():
            # Check for codes specifically defined in the JSON under PI~EXD* umbrella (e.g., PIRELBOEXD, DTISACRAEXD)
            # This logic must be robust to specific codes, not just prefixes, as per JSON
            if header.startswith('PI') and header.endswith('EXD') or \
               header.startswith('DTI') and header.endswith('EXD') or \
               header.startswith('PI') and header.endswith('D'): # Including PI~D for stage 3/4/unstageable PU codes themselves
                if self.code_sets.get(header):
                    self.pi_exd_codes_for_principal_exclusion.update(self.code_sets[header])

        # PSI_15 Specific Organ System Mappings
        self.organ_system_mappings: Dict[str, Dict[str, str]] = {
            'spleen': {'dx_codes': 'SPLEEN15D', 'proc_codes': 'SPLEEN15P'},
            'adrenal': {'dx_codes': 'ADRENAL15D', 'proc_codes': 'ADRENAL15P'},
            'vessel': {'dx_codes': 'VESSEL15D', 'proc_codes': 'VESSEL15P'},
            'diaphragm': {'dx_codes': 'DIAPHR15D', 'proc_codes': 'DIAPHR15P'},
            'gastrointestinal': {'dx_codes': 'GI15D', 'proc_codes': 'GI15P'},
            'genitourinary': {'dx_codes': 'GU15D', 'proc_codes': 'GU15P'},
        }
        # Consolidate all PSI_15 injury DX codes for easier lookup
        self.all_psi15_injury_dx_codes: Set[str] = set()
        for system_map in self.organ_system_mappings.values():
            # Populate with actual codes from code_sets, not just the code set names
            self.all_psi15_injury_dx_codes.update(self.code_sets.get(system_map['dx_codes'], set()))

        # PSI_04 Strata Definitions (for internal use)
        # Ordered by priority as per JSON (1 is highest)
        self.psi04_strata_priority = [
            'STRATUM_SHOCK',
            'STRATUM_SEPSIS',
            'STRATUM_PNEUMONIA',
            'STRATUM_GI_HEMORRHAGE',
            'STRATUM_DVT_PE'
        ]

    def _load_code_sets(self, codes_source_path: str) -> Dict[str, Set[str]]:
        """
        Loads code reference sets from a JSON file.
        The JSON file is expected to have a structure like:
        {"CODE_SET_NAME_1": ["code1", "code2", ...], "CODE_SET_NAME_2": ["codeA", "codeB", ...]}

        Args:
            codes_source_path (str): Path to the JSON file containing code sets.

        Returns:
            dict: A dictionary where keys are code set names and values are sets of codes.
        """
        code_sets: Dict[str, Set[str]] = {}
        try:
            with open(codes_source_path, 'r') as f:
                data = json.load(f)
                for code_set_name, codes_list in data.items():
                    if not isinstance(codes_list, list):
                        print(f"Warning: Code set '{code_set_name}' in '{codes_source_path}' is not a list. Skipping.")
                        continue
                    code_sets[code_set_name] = set(codes_list)
                    if not codes_list:
                        print(f"Warning: Code set '{code_set_name}' is empty. Ensure all required code sets have values.")
            print(f"Successfully loaded {len(code_sets)} code sets from {codes_source_path}.")
        except FileNotFoundError:
            print(f"Error: Code sets file not found at {codes_source_path}. Initializing with empty code sets.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {codes_source_path}. Check file format.")
        except Exception as e:
            print(f"Error loading code sets from {codes_source_path}: {e}")
        return code_sets

    def _load_psi_definitions(self, psi_definitions_path: str) -> Dict[str, Any]:
        """
        Loads PSI definitions from a JSON file.

        Args:
            psi_definitions_path (str): Path to the JSON file containing PSI definitions.

        Returns:
            dict: A dictionary containing PSI definitions.
        """
        try:
            with open(psi_definitions_path, 'r') as f:
                psi_data: Dict[str, Any] = json.load(f)
                return psi_data.get('data', {})
        except FileNotFoundError:
            print(f"Error: PSI definitions file not found at {psi_definitions_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {psi_definitions_path}")
            return {}
        except Exception as e:
            print(f"Error loading PSI definitions: {e}")
            return {}

    def _parse_date_string(self, date_str: Any, time_str: Any = None, encounter_id: Optional[str] = None) -> pd.Timestamp:
        """
        Parses a date string (and optional time string) into a datetime object.
        Handles missing or invalid formats gracefully by returning pd.NaT.
        """
        if pd.isna(date_str):
            return pd.NaT
        try:
            # Attempt to parse as date first
            dt_obj = datetime.strptime(str(date_str).strip(), '%Y-%m-%d')
            if pd.notna(time_str) and str(time_str).strip():
                # Pad time string with leading zeros if needed (e.g., '100' -> '0100')
                time_str_padded = str(int(time_str)).zfill(4) if pd.api.types.is_number(time_str) else str(time_str).strip().zfill(4)
                dt_obj = datetime.strptime(f"{date_str} {time_str_padded}", '%Y-%m-%d %H%M')
            return pd.Timestamp(dt_obj)
        except (ValueError, TypeError) as e:
            if encounter_id:
                print(f"Warning: Could not parse date/time for EncounterID {encounter_id}: '{date_str}' '{time_str}' - {e}")
            else:
                print(f"Warning: Could not parse date/time '{date_str}' '{time_str}' - {e}")
            return pd.NaT

    def _get_admission_discharge_dates(self, row: pd.Series) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Extracts and parses Admission_Date and Discharge_Date from a row.
        """
        admission_date: pd.Timestamp = self._parse_date_string(row.get('Admission_Date'), encounter_id=row.get('EncounterID'))
        discharge_date: pd.Timestamp = self._parse_date_string(row.get('Discharge_Date'), encounter_id=row.get('EncounterID'))
        return admission_date, discharge_date

    def _get_all_diagnoses(self, row: pd.Series) -> List[Dict[str, Optional[str]]]:
        """
        Extracts all diagnosis codes and their POA statuses from a row,
        correctly mapping Pdx to POA1, DX1 to POA2, etc.
        """
        diagnoses: List[Dict[str, Optional[str]]] = []

        # --- FIX: Principal diagnosis: Pdx + POA1 ---
        pdx_code = row.get('Pdx')
        pdx_poa = row.get('POA1')
        if pd.notna(pdx_code):
            diagnoses.append({'code': str(pdx_code), 'poa': str(pdx_poa) if pd.notna(pdx_poa) else None})
        # --- DEBUG: Print extracted Pdx and POA1 ---
        # print(f"DEBUG: Pdx: {pdx_code}, POA1: {pdx_poa}")

        # --- FIX: Secondary diagnoses: DX1-DX25 + POA2-POA26 ---
        for i in range(1, 26):  # DX1 to DX25
            dx_code = row.get(f'DX{i}')
            poa_status_col_name = f'POA{i+1}' # POA for DX_i is POA_(i+1)
            poa_status = row.get(poa_status_col_name) if poa_status_col_name in row else None

            if pd.notna(dx_code):
                diagnoses.append({'code': str(dx_code), 'poa': str(poa_status) if pd.notna(poa_status) else None})
            # --- DEBUG: Print extracted secondary DX and POA ---
            # if pd.notna(dx_code):
            #     print(f"DEBUG: DX{i}: {dx_code}, {poa_status_col_name}: {poa_status}")

        return diagnoses

    def _get_all_procedures(self, row: pd.Series) -> List[Dict[str, pd.Timestamp]]:
        """Extracts all procedure codes and their dates from a row."""
        procedures: List[Dict[str, pd.Timestamp]] = []
        for i in range(1, 11): # Proc1 to Proc10
            proc_code = row.get(f'Proc{i}')
            proc_date = self._parse_date_string(row.get(f'Proc{i}_Date'), row.get(f'Proc{i}_Time'), row.get('EncounterID'))

            if pd.notna(proc_code):
                procedures.append({'code': str(proc_code), 'date': proc_date})
        return procedures

    def _calculate_days_diff(self, date1: pd.Timestamp, date2: pd.Timestamp) -> Optional[int]:
        """
        Calculates the difference in days between two datetime objects.
        Returns None if either date is NaT.
        """
        if pd.isna(date1) or pd.isna(date2):
            return None
        return (date2 - date1).days

    def _get_first_procedure_date_by_code_set(self, procedures: List[Dict[str, pd.Timestamp]], code_set_name: str) -> pd.Timestamp:
        """
        Finds the earliest date among procedures belonging to a specific code set.
        Returns pd.NaT if no procedures from the set are found or dates are missing.
        """
        min_date = pd.NaT
        target_codes = self.code_sets.get(code_set_name, set())

        if not target_codes:
            return pd.NaT # No codes defined for this set

        for proc_entry in procedures:
            if proc_entry['code'] in target_codes and pd.notna(proc_entry['date']):
                if pd.isna(min_date) or proc_entry['date'] < min_date:
                    min_date = proc_entry['date']
        return min_date

    def _get_latest_procedure_date_by_code_set(self, procedures: List[Dict[str, pd.Timestamp]], code_set_name: str) -> pd.Timestamp:
        """
        Finds the latest date among procedures belonging to a specific code set.
        Returns pd.NaT if no procedures from the set are found or dates are missing.
        """
        max_date = pd.NaT
        target_codes = self.code_sets.get(code_set_name, set())

        if not target_codes:
            return pd.NaT # No codes defined for this set

        for proc_entry in procedures:
            if proc_entry['code'] in target_codes and pd.notna(proc_entry['date']):
                if pd.isna(max_date) or proc_entry['date'] > max_date:
                    max_date = proc_entry['date']
        return max_date

    def _check_procedure_timing(self, procedures: List[Dict[str, pd.Timestamp]], ref_date: pd.Timestamp, target_code_set_name: str, min_days: Optional[int] = None, max_days: Optional[int] = None, inclusive_min: bool = True, inclusive_max: bool = True) -> bool:
        """
        Checks if any procedure from a target_code_set_name falls within a specified
        time window relative to a reference date.

        Args:
            procedures (list): List of procedure dictionaries (from _get_all_procedures).
            ref_date (datetime): The reference date (e.g., admission date, first OR procedure date).
            target_code_set_name (str): The name of the code set for procedures to check.
            min_days (int, optional): Minimum number of days after ref_date. Defaults to None.
            max_days (int, optional): Maximum number of days after ref_date. Defaults to None.
            inclusive_min (bool): If True, min_days is inclusive (>=). If False, exclusive (>).
            inclusive_max (bool): If True, max_days is inclusive (<=). If False, exclusive (<).

        Returns:
            bool: True if a qualifying procedure is found, False otherwise.
        """
        if pd.isna(ref_date):
            return False # Cannot check timing without a reference date

        target_codes = self.code_sets.get(target_code_set_name, set())
        if not target_codes:
            return False # No codes defined for this set

        for proc_entry in procedures:
            proc_code = proc_entry['code']
            proc_date = proc_entry['date']

            if proc_code in target_codes and pd.notna(proc_date):
                days_diff = self._calculate_days_diff(ref_date, proc_date)
                if days_diff is None:
                    continue # Skip if date calculation failed for this procedure

                # Apply timing window logic
                is_within_window = True
                if min_days is not None:
                    if inclusive_min and days_diff < min_days:
                        is_within_window = False
                    elif not inclusive_min and days_diff <= min_days:
                        is_within_window = False
                if max_days is not None:
                    if inclusive_max and days_diff > max_days:
                        is_within_window = False
                    elif not inclusive_max and days_diff >= max_days:
                        is_within_window = False

                if is_within_window:
                    return True # Found a procedure within the window
        return False

    def _get_organ_system_from_code(self, code: str, is_dx: bool = True) -> Optional[str]:
        """
        Determines the organ system associated with a given diagnosis or procedure code.
        """
        for system, codes in self.organ_system_mappings.items():
            code_set_name = codes['dx_codes'] if is_dx else codes['proc_codes']
            if code_set_name in self.code_sets and code in self.code_sets[code_set_name]:
                return system
        return None

    def _assign_psi13_risk_category(self, all_diagnoses: List[Dict[str, Optional[str]]], all_procedures: List[Dict[str, pd.Timestamp]]) -> str:
        """
        Assigns a risk category for PSI_13 (Postoperative Sepsis) based on immune function severity.
        Mutually exclusive assignment: highest priority category wins.

        Categories (Priority 1-4):
        1. severe_immune_compromise (SEVEREIMMUNEDX, SEVEREIMMUNEPROC)
        2. moderate_immune_compromise (MODERATEIMMUNEDX, MODERATEIMMUNEPROC)
        3. malignancy_with_treatment (CANCEID + CHEMORADTXPROC)
        4. baseline_risk (default)
        """
        # Priority 1: Severe Immune Compromise
        for dx_entry in all_diagnoses:
            if dx_entry['code'] in self.code_sets.get('SEVEREIMMUNEDX', set()):
                return "severe_immune_compromise"
        for proc_entry in all_procedures:
            if proc_entry['code'] in self.code_sets.get('SEVEREIMMUNEPROC', set()):
                return "severe_immune_compromise"

        # Priority 2: Moderate Immune Compromise
        for dx_entry in all_diagnoses:
            if dx_entry['code'] in self.code_sets.get('MODERATEIMMUNEDX', set()):
                return "moderate_immune_compromise"
        for proc_entry in all_procedures:
            if proc_entry['code'] in self.code_sets.get('MODERATEIMMUNEPROC', set()):
                return "moderate_immune_compromise"

        # Priority 3: Malignancy with Treatment
        has_cancer_dx = any(dx_entry['code'] in self.code_sets.get('CANCEID', set()) for dx_entry in all_diagnoses)
        has_chemorad_proc = any(proc_entry['code'] in self.code_sets.get('CHEMORADTXPROC', set()) for proc_entry in all_procedures)
        if has_cancer_dx and has_chemorad_proc:
            return "malignancy_with_treatment"

        # Priority 4: Baseline Risk
        return "baseline_risk"

    def _assign_psi15_risk_category(self, all_procedures: List[Dict[str, pd.Timestamp]], index_date: pd.Timestamp) -> str:
        """
        Assigns a risk category for PSI_15 based on procedure complexity on the index date.

        Categories:
        - high_complexity (PCLASSHIGH procedures on index_date)
        - moderate_complexity (PCLASSMODERATE procedures on index_date, if not high)
        - low_complexity (default)
        """
        if pd.isna(index_date):
            return "low_complexity" # Cannot determine complexity without index date

        procedures_on_index_date = [
            proc_entry for proc_entry in all_procedures
            if pd.notna(proc_entry['date']) and proc_entry['date'].date() == index_date.date()
        ]

        # Check for high complexity procedures
        for proc_entry in procedures_on_index_date:
            if proc_entry['code'] in self.code_sets.get('PCLASSHIGH', set()):
                return "high_complexity"

        # Check for moderate complexity procedures
        for proc_entry in procedures_on_index_date:
            if proc_entry['code'] in self.code_sets.get('PCLASSMODERATE', set()):
                return "moderate_complexity"

        return "low_complexity"

    def _assign_psi14_stratum(self, all_procedures: List[Dict[str, pd.Timestamp]]) -> str:
        """
        Assigns a stratum for PSI_14 (Postoperative Wound Dehiscence) based on the type of
        abdominopelvic surgery (open vs. non-open).
        Open approach takes priority.
        """
        has_open_proc = any(p['code'] in self.code_sets.get('ABDOMIPOPEN', set()) for p in all_procedures)
        has_non_open_proc = any(p['code'] in self.code_sets.get('ABDOMIPOTHER', set()) for p in all_procedures)

        if has_open_proc:
            return "open_approach"
        elif has_non_open_proc:
            return "non_open_approach"
        else:
            return "unknown_approach" # Should ideally be caught by denominator inclusion

    def _check_base_exclusions(self, row: pd.Series, psi_code: str) -> Optional[Tuple[str, str]]:
        """
        Applies base exclusion logic common to many PSIs.
        This includes age, MDC, and general data quality checks.

        Args:
            row (pd.Series): A single row of patient encounter data.
            psi_code (str): The code of the PSI being evaluated (e.g., 'PSI_02').

        Returns:
            tuple: (status, reason) if excluded, None otherwise.
        """
        # Retrieve PSI-specific definitions for exclusions
        psi_def = self.psi_definitions.get(psi_code, {})
        # Note: 'indicator' is nested inside the PSI definition in the JSON, get() is safer
        population_type = psi_def.get('indicator', {}).get('population_type')

        # Dynamically determine required fields based on PSI definition's data_quality rules
        # --- FIX: Updated required_fields list to match prompt's headers ---
        required_fields: List[str] = ['EncounterID', 'AGE', 'SEX', 'MS-DRG', 'MDC', 'Pdx', 'POA1']

        # Add fields explicitly marked as required in data_quality section of PSI definition
        for excl_group in psi_def.get('exclusion_criteria', []):
            if excl_group.get('category') == 'data_quality':
                for rule in excl_group.get('rules', []):
                    if rule.get('description') == 'Missing required fields' and 'fields' in rule:
                        for field_def in rule['fields']:
                            if field_def['name'] not in required_fields: # Avoid duplicates
                                required_fields.append(field_def['name'])

        # Add Admission_Date and Discharge_Date if timing is required for the PSI
        if psi_def.get('indicator', {}).get('requires_procedure_timing') or \
           psi_def.get('indicator', {}).get('requires_time_windows'):
            if 'Admission_Date' not in required_fields:
                required_fields.append('Admission_Date')
            if 'Discharge_Date' not in required_fields:
                required_fields.append('Discharge_Date')

        # Add Length_of_stay if required for the PSI
        if psi_def.get('indicator', {}).get('requires_minimum_los'):
            if 'Length_of_stay' not in required_fields:
                required_fields.append('Length_of_stay')

        for field in required_fields:
            if field not in row or pd.isna(row.get(field)):
                return "Exclusion", f"Data Exclusion: Missing required field '{field}'"

        # Age exclusion logic based on population type
        age = row.get('AGE')
        if pd.isna(age) or not isinstance(age, (int, float)):
             return "Exclusion", "Data Exclusion: Invalid or missing 'AGE'"
        age = int(age) # Convert to int after checking for NaN/type

        if population_type == 'adult':
            if age < 18:
                return "Exclusion", "Population Exclusion: Age < 18"
        elif population_type == 'newborn_only':
            # For newborn PSIs, age < 18 is expected, so no exclusion here.
            pass
        elif population_type in ['maternal_obstetric', 'elective_surgical_only', 'surgical_only', 'abdominopelvic_surgical', 'medical_and_surgical']:
            # For these, age >= 18 is generally expected, but obstetric patients can be any age.
            # Check for specific age criteria in PSI definition, otherwise apply general age < 18.
            is_obstetric_any_age_allowed = False
            # Check if current encounter is explicitly identified as obstetric (MDC 14 principal DX)
            principal_dx_code = row.get('Pdx') # --- FIX: Use Pdx for principal diagnosis check ---
            mdc = row.get('MDC')
            is_obstetric_mdc14 = (pd.notna(mdc) and int(mdc) == 14) and \
                                 (principal_dx_code and str(principal_dx_code) in self.code_sets.get('MDC14PRINDX', set()))

            if is_obstetric_mdc14:
                # Check JSON to see if 'obstetric patients of any age' is specified for this PSI
                # This is typically found in denominator inclusion for maternal_obstetric type.
                for incl_crit in psi_def.get('denominator', {}).get('inclusion_criteria', []):
                    if incl_crit.get('type') == 'age' and incl_crit.get('description') == 'obstetric patients of any age':
                        is_obstetric_any_age_allowed = True
                        break
            if not is_obstetric_any_age_allowed and age < 18:
                 return "Exclusion", "Population Exclusion: Age < 18"


        # MDC 15 (Newborn) exclusion logic based on population type
        mdc = row.get('MDC')
        if pd.notna(mdc):
            try:
                mdc_int = int(mdc)
                if mdc_int == 15:
                    # Check if Pdx is in MDC15PRINDX (specific to principal DX rule)
                    pdx = row.get('Pdx') # --- FIX: Use Pdx for principal diagnosis check ---
                    if pd.notna(pdx) and str(pdx) in self.code_sets.get('MDC15PRINDX', set()):
                        if population_type != 'newborn_only': # Only exclude if not a newborn-specific PSI
                            return "Exclusion", "Population Exclusion: MDC 15 - Newborn (principal dx in MDC15PRINDX)"
            except ValueError:
                return "Exclusion", "Data Exclusion: Invalid MDC value"

        # MDC 14 (Obstetric) exclusion logic based on population type
        if pd.notna(mdc):
            try:
                mdc_int = int(mdc)
                if mdc_int == 14:
                    # Check if Pdx is in MDC14PRINDX (specific to principal DX rule)
                    pdx = row.get('Pdx') # --- FIX: Use Pdx for principal diagnosis check ---
                    if pd.notna(pdx) and str(pdx) in self.code_sets.get('MDC14PRINDX', set()):
                        if population_type != 'maternal_obstetric': # Only exclude if not an obstetric-specific PSI
                            return "Exclusion", "Population Exclusion: MDC 14 - Obstetric (principal dx in MDC14PRINDX)"
            except ValueError:
                return "Exclusion", "Data Exclusion: Invalid MDC value"

        # Ungroupable DRG exclusion (common)
        drg = str(row.get('MS-DRG'))
        if pd.notna(drg) and drg == '999':
            return "Exclusion", "Data Exclusion: DRG is ungroupable (999)"

        return None # No base exclusion met

    def evaluate_psi(self, row: pd.Series, psi_code: str) -> Tuple[str, str, str, Dict[str, Any]]:
        """
        Evaluates a single patient encounter against a specific PSI.
        This method will call the PSI-specific evaluation function.

        Args:
            row (pd.Series): A single row of patient encounter data.
            psi_code (str): The code of the PSI to evaluate (e.g., 'PSI_02').

        Returns:
            tuple: (status, reason, psi_category, details)
        """
        # Apply base exclusions first
        base_exclusion_result = self._check_base_exclusions(row, psi_code)
        if base_exclusion_result:
            status, reason = base_exclusion_result
            return status, reason, psi_code, {}

        # Call the specific PSI evaluation function
        eval_func_name = f"evaluate_{psi_code.lower()}"
        if hasattr(self, eval_func_name):
            eval_func = getattr(self, eval_func_name)
            try:
                status, reason = eval_func(row, self.code_sets)
                return status, reason, psi_code, {} # Details can be expanded by specific PSI functions
            except Exception as e:
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                return "Error", f"An error occurred during PSI evaluation: {e}", psi_code, {}
        else:
            return "Not Implemented", f"Evaluation logic for {psi_code} not found.", psi_code, {}

    def evaluate_psi02(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_02: Death Rate in Low-Mortality DRGs.

        Denominator: Age >=18, low-mortality DRG (LOWMODR codes), exclude trauma/cancer/immunocompromised,
                     exclude transfers to acute care, exclude hospice admissions, exclude MDC 15.
        Numerator: Death disposition (DISP=20) among eligible cases.
        POA Logic: POA status does not affect exclusions for TRAUMID, CANCEID, IMMUNID.
                   Numerator is based on Discharge_Disposition, so POA is not applicable.
        """
        # Denominator Inclusion: Low-mortality DRG
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('LOWMODR', set()):
            return "Exclusion", "Denominator Exclusion: Not a low-mortality DRG"

        # Denominator Exclusions (Clinical - POA does not matter for these exclusions as per JSON description)
        all_diagnoses = self._get_all_diagnoses(row)
        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            if dx_code in appendix.get('TRAUMID', set()):
                return "Exclusion", f"Denominator Exclusion: Trauma diagnosis present ({dx_code})"
            if dx_code in appendix.get('CANCEID', set()):
                return "Exclusion", f"Denominator Exclusion: Cancer diagnosis present ({dx_code})"
            if dx_code in appendix.get('IMMUNID', set()):
                return "Exclusion", f"Denominator Exclusion: Immunocompromised diagnosis present ({dx_code})"

        all_procedures = self._get_all_procedures(row)
        for proc_entry in all_procedures:
            proc_code = proc_entry['code']
            if proc_code in appendix.get('IMMUNIP', set()):
                return "Exclusion", f"Denominator Exclusion: Immunocompromising procedure present ({proc_code})"

        # Denominator Exclusions (Admission/Transfer)
        point_of_origin = row.get('POINTOFORIGINUB04')
        if pd.notna(point_of_origin) and str(point_of_origin) == 'F':
            return "Exclusion", "Denominator Exclusion: Admission from hospice facility"

        # PSI_02 specific transfer exclusion (Discharge_Disposition = 2)
        discharge_disposition = row.get('Discharge_Disposition')
        if pd.notna(discharge_disposition) and int(discharge_disposition) == 2:
            return "Exclusion", "Population Exclusion: Transfer to acute care facility (Discharge_Disposition=2)"

        # Numerator Check
        if pd.notna(discharge_disposition) and int(discharge_disposition) == 20:
            return "Inclusion", "Inclusion: Death disposition (DISP=20)"
        else:
            # Case is in denominator but not numerator
            return "Exclusion", "Exclusion: Not a death disposition (DISP!=20) but in denominator"

    def evaluate_psi03(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_03: Pressure Ulcer Rate.

        Denominator: Surgical/medical DRG (SURGI2R/MEDIC2R), Age >=18, LOS >= 3 days.
        Numerator: Stage 3/4 pressure ulcer NOT POA, AND NOT excluded by DTI POA at same anatomic site,
                   OR unspecified site pressure ulcer NOT POA.
        Exclusions: Principal DX of pressure ulcer/DTI, severe burns, exfoliative skin disorders.
        """
        # Denominator Inclusion: Surgical or Medical DRG
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()) and drg not in appendix.get('MEDIC2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical or medical MS-DRG"

        # Denominator Inclusion: Length of Stay >= 3 days
        los = row.get('Length_of_stay')
        if pd.isna(los) or int(los) < 3:
            return "Exclusion", "Denominator Exclusion: Length of stay less than 3 days or missing"

        all_diagnoses = self._get_all_diagnoses(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Exclusions (Clinical)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        # Exclusion: Principal diagnosis of pressure ulcer stage 3/4/unstageable or deep tissue injury
        if principal_dx_code in self.pi_exd_codes_for_principal_exclusion:
            return "Exclusion", f"Denominator Exclusion: Principal diagnosis is pressure ulcer/DTI ({principal_dx_code})"

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            # Exclusion: Severe burns or exfoliative skin disorders
            if dx_code in appendix.get('BURNDX', set()):
                return "Exclusion", f"Denominator Exclusion: Severe burn diagnosis present ({dx_code})"
            if dx_code in appendix.get('EXFOLIATXD', set()):
                return "Exclusion", f"Denominator Exclusion: Exfoliative skin disorder diagnosis present ({dx_code})"

        # Numerator Logic:
        has_qualifying_pu = False
        for pu_dx_entry in all_diagnoses:
            pu_dx_code = pu_dx_entry['code']
            pu_poa_status = pu_dx_entry['poa']

            # Check if it's a Stage 3/4 or unstageable pressure ulcer AND not POA (N/U/W/null)
            if pu_poa_status in ['N', 'U', 'W', None] or pd.isna(pu_poa_status):
                # Handle unspecified site pressure ulcers (automatically qualify)
                if pu_dx_code in self.unspecified_pu_codes:
                    has_qualifying_pu = True
                    break # Found a qualifying PU, no further checks needed for this patient

                # Handle specific anatomic site pressure ulcers
                if pu_dx_code in self.all_specific_pu_codes:
                    # Get the corresponding DTI exclusion code set name for this anatomic site
                    dti_ex_code_set_name = self.anatomic_site_map.get(pu_dx_code)
                    if dti_ex_code_set_name:
                        # Check if a DTI for the SAME anatomic site is POA='Y'
                        is_dti_poa_same_site = False
                        # Ensure dti_ex_code_set_name actually has codes defined
                        dti_exclusion_codes = appendix.get(dti_ex_code_set_name, set())
                        if dti_exclusion_codes:
                            for dti_dx_entry in all_diagnoses:
                                if dti_dx_entry['code'] in dti_exclusion_codes and dti_dx_entry['poa'] == 'Y':
                                    is_dti_poa_same_site = True
                                    break
                        if not is_dti_poa_same_site:
                            has_qualifying_pu = True
                            break # Found a qualifying PU, no further checks needed for this patient

        if has_qualifying_pu:
            return "Inclusion", "Inclusion: Hospital-acquired pressure ulcer (Stage 3/4 or Unstageable)"
        else:
            return "Exclusion", "Exclusion: No qualifying hospital-acquired pressure ulcer identified"

    def _check_psi04_stratum_criteria(self, stratum_name: str, row: pd.Series, appendix: Dict[str, Set[str]],
                                      all_diagnoses: List[Dict[str, Optional[str]]],
                                      all_procedures: List[Dict[str, pd.Timestamp]],
                                      first_or_proc_date: pd.Timestamp) -> bool:
        """
        Helper function to check if a patient qualifies for a specific PSI_04 stratum.
        Assumes general denominator criteria (DRG, ORPROC presence) are already met.
        """
        psi04_def = self.psi_definitions.get('PSI_04', {})
        strata_defs = psi04_def.get('strata_definitions', {})
        stratum_data = strata_defs.get(stratum_name, {})

        # Check stratum inclusion criteria
        meets_inclusion = False
        # Inclusion criteria are typically diagnoses secondary and not POA, or procedures with timing
        principal_dx_code = all_diagnoses[0]['code'] if all_diagnoses else None

        if stratum_name == 'STRATUM_SHOCK':
            # Inclusion: Secondary FTR5DX* (not POA) OR any FTR5PR* procedure (same day as or after first OR procedure)
            has_secondary_ftr5dx = any(dx_entry['code'] in appendix.get('FTR5DX', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])) for dx_entry in all_diagnoses[1:])
            has_ftr5pr_after_or = self._check_procedure_timing(all_procedures, first_or_proc_date, 'FTR5PR', min_days=0, inclusive_min=True)
            meets_inclusion = has_secondary_ftr5dx or has_ftr5pr_after_or
        elif stratum_name == 'STRATUM_SEPSIS':
            # Inclusion: Secondary FTR4DX* (not POA)
            meets_inclusion = any(dx_entry['code'] in appendix.get('FTR4DX', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])) for dx_entry in all_diagnoses[1:])
        elif stratum_name == 'STRATUM_PNEUMONIA':
            # Inclusion: Secondary FTR3DX* (not POA)
            meets_inclusion = any(dx_entry['code'] in appendix.get('FTR3DX', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])) for dx_entry in all_diagnoses[1:])
        elif stratum_name == 'STRATUM_GI_HEMORRHAGE':
            # Inclusion: Secondary FTR6DX* (not POA)
            meets_inclusion = any(dx_entry['code'] in appendix.get('FTR6DX', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])) for dx_entry in all_diagnoses[1:])
        elif stratum_name == 'STRATUM_DVT_PE':
            # Inclusion: Secondary FTR2DXB* (not POA)
            meets_inclusion = any(dx_entry['code'] in appendix.get('FTR2DXB', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])) for dx_entry in all_diagnoses[1:])

        if not meets_inclusion:
            return False

        # Check stratum exclusion criteria
        mdc = row.get('MDC')
        mdc_int = int(mdc) if pd.notna(mdc) else None

        for excl_rule_str in stratum_data.get('exclusion_criteria', []):
            # Parse exclusion rules from string descriptions (simplified for this context)
            # This is a brittle approach; a more robust JSON rule parser would be better.
            if "Principal diagnosis of" in excl_rule_str:
                code_ref_name_part = excl_rule_str.split("Principal diagnosis of ")[1].split(" ")[0]
                code_ref_name = code_ref_name_part.replace("(", "").replace(")", "").replace("*", "")
                if principal_dx_code and principal_dx_code in appendix.get(code_ref_name, set()):
                    return False
            elif "Any diagnosis of" in excl_rule_str:
                code_ref_name_part = excl_rule_str.split("Any diagnosis of ")[1].split(" ")[0]
                code_ref_name = code_ref_name_part.replace("(", "").replace(")", "").replace("*", "")
                if any(dx_entry['code'] in appendix.get(code_ref_name, set()) for dx_entry in all_diagnoses):
                    return False
            elif "Esophageal varices with bleeding" in excl_rule_str:
                has_ftr6gv = any(dx_entry['code'] in appendix.get('FTR6GV', set()) for dx_entry in all_diagnoses)
                has_ftr6qd = any(dx_entry['code'] in appendix.get('FTR6QD', set()) for dx_entry in all_diagnoses)
                if has_ftr6gv and has_ftr6qd:
                    return False
            elif "MDC 4 (Respiratory)" in excl_rule_str and mdc_int == 4:
                return False
            elif "MDC 5 (Circulatory)" in excl_rule_str and mdc_int == 5:
                return False
            elif "MDC 6 (Digestive)" in excl_rule_str and mdc_int == 6:
                return False
            elif "MDC 7 (Hepatobiliary)" in excl_rule_str and mdc_int == 7:
                return False
            elif "Any procedure for lung cancer" in excl_rule_str:
                if any(proc_entry['code'] in appendix.get('LUNGCIP', set()) for proc_entry in all_procedures):
                    return False

        return True # Meets inclusion and no stratum-specific exclusions

    def evaluate_psi04(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_04: Death Rate among Surgical Inpatients with Serious Treatable Complications.

        Denominator: Surgical discharges (SURGI2R), age 18-89 (or obstetric any age), with OR procedures,
                     and elective admission OR OR procedure within 2 days of admission,
                     AND a serious treatable complication from one of the 5 strata.
        Numerator: Death disposition (DISP=20) among eligible cases.
        """
        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        admission_date, _ = self._get_admission_discharge_dates(row)

        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Surgical DRG
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical MS-DRG"

        # Denominator Inclusion: Age 18-89 OR obstetric patient of any age
        age = row.get('AGE')
        mdc = row.get('MDC')
        principal_dx_code = all_diagnoses[0]['code'] if all_diagnoses else None
        is_obstetric_mdc14 = (pd.notna(mdc) and int(mdc) == 14) and \
                             (principal_dx_code and str(principal_dx_code) in self.code_sets.get('MDC14PRINDX', set()))

        if not is_obstetric_mdc14:
            if pd.isna(age) or not (18 <= int(age) <= 89):
                return "Exclusion", "Population Exclusion: Age not 18-89 and not an obstetric patient"
        # If it is an obstetric patient (MDC14PRINDX), any age is allowed, so no age exclusion here.

        # Denominator Inclusion: At least one OR procedure
        first_or_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ORPROC')
        if pd.isna(first_or_proc_date):
            return "Exclusion", "Denominator Exclusion: No qualifying OR procedure found"

        # Denominator Inclusion: Admission Timing (Elective OR OR procedure within 2 days of admission)
        admission_type = row.get('ATYPE')
        is_elective_admission = (pd.notna(admission_type) and int(admission_type) == 3)
        or_proc_within_2_days = False
        if pd.notna(admission_date) and pd.notna(first_or_proc_date):
            days_diff = self._calculate_days_diff(admission_date, first_or_proc_date)
            if days_diff is not None and days_diff <= 2:
                or_proc_within_2_days = True

        if not (is_elective_admission or or_proc_within_2_days):
            return "Exclusion", "Denominator Exclusion: Not elective admission and first OR not within 2 days of admission"

        # Apply overall exclusions (from PSI_04 JSON's 'overall_exclusions' category)
        # Transfer to acute care facility (DISP=2)
        discharge_disposition = row.get('Discharge_Disposition')
        if pd.notna(discharge_disposition) and int(discharge_disposition) == 2:
            return "Exclusion", "Overall Exclusion: Transfer to acute care facility (Discharge_Disposition=2)"

        # Admission from hospice facility (POINTOFORIGINUB04='F')
        point_of_origin = row.get('POINTOFORIGINUB04')
        if pd.notna(point_of_origin) and str(point_of_origin) == 'F':
            return "Exclusion", "Overall Exclusion: Admission from hospice facility"

        # Newborn and neonatal discharges (MDC 15 principal diagnosis)
        if pd.notna(mdc) and int(mdc) == 15 and principal_dx_code and str(principal_dx_code) in self.code_sets.get('MDC15PRINDX', set()):
            return "Exclusion", "Overall Exclusion: MDC 15 - Newborn (principal dx in MDC15PRINDX)"


        # Identify qualifying complication stratum (highest priority wins)
        assigned_stratum: Optional[str] = None
        for stratum_name in self.psi04_strata_priority:
            if self._check_psi04_stratum_criteria(stratum_name, row, appendix, all_diagnoses, all_procedures, first_or_proc_date):
                assigned_stratum = stratum_name
                break # Assign to highest priority stratum found

        if assigned_stratum is None:
            return "Exclusion", "Exclusion: No serious treatable complication identified"

        # Numerator Check: Death disposition (DISP=20)
        if pd.notna(discharge_disposition) and int(discharge_disposition) == 20:
            return "Inclusion", f"Inclusion: Death among surgical inpatients with {assigned_stratum.replace('STRATUM_', '').replace('_', ' ')}"
        else:
            return "Exclusion", f"Exclusion: Not a death disposition (DISP!=20) but in {assigned_stratum.replace('STRATUM_', '').replace('_', ' ')} denominator"

    def evaluate_psi05(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_05: Retained Surgical Item or Unretrieved Device Fragment Count.

        Denominator: Surgical or Medical DRG (SURGI2R/MEDIC2R), Age >=18.
        Numerator: FOREIID codes (secondary, not POA).
        This is a COUNT indicator, meaning it identifies cases for counting, not a rate calculation within this function.
        """
        # Denominator Inclusion: Surgical or Medical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()) and drg not in appendix.get('MEDIC2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical or medical MS-DRG"

        all_diagnoses = self._get_all_diagnoses(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Exclusions: Principal DX of FOREIID or Secondary DX of FOREIID POA='Y'
        principal_dx_code = all_diagnoses[0]['code']
        if principal_dx_code and principal_dx_code in appendix.get('FOREIID', set()):
            return "Exclusion", f"Denominator Exclusion: Principal diagnosis is retained surgical item ({principal_dx_code})"

        for dx_entry in all_diagnoses:
            if dx_entry['code'] in appendix.get('FOREIID', set()) and dx_entry['poa'] == 'Y':
                return "Exclusion", f"Denominator Exclusion: Retained surgical item diagnosis ({dx_entry['code']}) present on admission (POA=Y)"

        # Numerator Check: Secondary DX of FOREIID and not POA
        has_retained_item = False
        # Loop through all diagnoses, but exclude the principal for numerator check as per rules (secondary, not POA)
        # Note: All_diagnoses[0] is Pdx; all_diagnoses[1:] are DX1 to DX25 and their POAs
        for dx_entry in all_diagnoses[1:]: # Start from the first secondary diagnosis (DX1)
            if dx_entry['code'] in appendix.get('FOREIID', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_retained_item = True
                break

        if has_retained_item:
            return "Inclusion", "Inclusion: Retained surgical item or unretrieved device fragment (not POA)"
        else:
            return "Exclusion", "Exclusion: No qualifying retained surgical item or unretrieved device fragment found"

    def evaluate_psi06(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_06: Iatrogenic Pneumothorax.

        Denominator: Surgical or Medical DRG (SURGI2R/MEDIC2R), Age >=18.
        Numerator: IATROID codes (secondary, not POA).
        Exclusions: IATPTXD (principal or secondary POA='Y'), CTRAUMD (any), PLEURAD (any),
                    THORAIP procedures (any), CARDSIP procedures (any).
        """
        # Denominator Inclusion: Surgical or Medical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()) and drg not in appendix.get('MEDIC2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical or medical MS-DRG"

        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: IATPTXD (non-traumatic pneumothorax) if principal or secondary POA='Y'
            if dx_code in appendix.get('IATPTXD', set()):
                # Check if it's the principal diagnosis OR if it's secondary and POA='Y'
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Non-traumatic pneumothorax ({dx_code}) present on admission or as principal diagnosis"

            # Exclusion: CTRAUMD (chest trauma) any position
            if dx_code in appendix.get('CTRAUMD', set()):
                return "Exclusion", f"Denominator Exclusion: Chest trauma diagnosis present ({dx_code})"

            # Exclusion: PLEURAD (pleural effusion) any position
            if dx_code in appendix.get('PLEURAD', set()):
                return "Exclusion", f"Denominator Exclusion: Pleural effusion diagnosis present ({dx_code})"

        # Exclusions (Procedures)
        for proc_entry in all_procedures:
            proc_code = proc_entry['code']
            # Exclusion: THORAIP (thoracic surgery) procedures
            if proc_code in appendix.get('THORAIP', set()):
                return "Exclusion", f"Denominator Exclusion: Thoracic surgery procedure present ({proc_code})"
            # Exclusion: CARDSIP (trans-pleural cardiac) procedures
            if proc_code in appendix.get('CARDSIP', set()):
                return "Exclusion", f"Denominator Exclusion: Trans-pleural cardiac procedure present ({proc_code})"

        # Numerator Check: IATROID (iatrogenic pneumothorax) secondary and not POA
        has_iatrogenic_pneumothorax = False
        # Start from the first secondary diagnosis (DX1) onwards
        for dx_entry in all_diagnoses[1:]:
            if dx_entry['code'] in appendix.get('IATROID', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_iatrogenic_pneumothorax = True
                break

        if has_iatrogenic_pneumothorax:
            return "Inclusion", "Inclusion: Iatrogenic pneumothorax (secondary, not POA)"
        else:
            return "Exclusion", "Exclusion: No qualifying iatrogenic pneumothorax found"

    def evaluate_psi07(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_07: Central Venous Catheter-Related Bloodstream Infection Rate.

        Denominator: Surgical or Medical DRG (SURGI2R/MEDIC2R), Age >=18, LOS >= 2 days.
        Numerator: IDTMC3D codes (secondary, not POA).
        Exclusions: CANCEID (any), IMMUNID (any), IMMUNIP procedures (any).
        """
        # Denominator Inclusion: Surgical or Medical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()) and drg not in appendix.get('MEDIC2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical or medical MS-DRG"

        # Denominator Inclusion: Length of Stay >= 2 days
        los = row.get('Length_of_stay')
        if pd.isna(los) or int(los) < 2:
            return "Exclusion", "Denominator Exclusion: Length of stay less than 2 days or missing"

        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            # Exclusion: CANCEID (cancer) any position
            if dx_code in appendix.get('CANCEID', set()):
                return "Exclusion", f"Denominator Exclusion: Cancer diagnosis present ({dx_code})"
            # Exclusion: IMMUNID (immunocompromised state) any position
            if dx_code in appendix.get('IMMUNID', set()):
                return "Exclusion", f"Denominator Exclusion: Immunocompromised diagnosis present ({dx_code})"

        # Exclusions (Procedures)
        for proc_entry in all_procedures:
            proc_code = proc_entry['code']
            # Exclusion: IMMUNIP (immunocompromising procedures)
            if proc_code in appendix.get('IMMUNIP', set()):
                return "Exclusion", f"Denominator Exclusion: Immunocompromising procedure present ({proc_code})"

        # Exclusions: Principal DX of IDTMC3D or Secondary DX of IDTMC3D POA='Y'
        if principal_dx_code and principal_dx_code in appendix.get('IDTMC3D', set()):
            return "Exclusion", f"Denominator Exclusion: Principal diagnosis is CVC-related BSI ({principal_dx_code})"

        for dx_entry in all_diagnoses:
            if dx_entry['code'] in appendix.get('IDTMC3D', set()) and dx_entry['poa'] == 'Y':
                return "Exclusion", f"Denominator Exclusion: CVC-related BSI diagnosis ({dx_code}) present on admission (POA=Y)"


        # Numerator Check: IDTMC3D (CVC-related BSI) secondary and not POA
        has_cvc_bsi = False
        # Start from the first secondary diagnosis (DX1) onwards
        for dx_entry in all_diagnoses[1:]:
            if dx_entry['code'] in appendix.get('IDTMC3D', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_cvc_bsi = True
                break

        if has_cvc_bsi:
            return "Inclusion", "Inclusion: Central venous catheter-related bloodstream infection (secondary, not POA)"
        else:
            return "Exclusion", "Exclusion: No qualifying CVC-related bloodstream infection found"


    def evaluate_psi08(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_08: In-Hospital Fall-Associated Fracture Rate.

        Denominator: Surgical or medical discharges for patients ages 18 years and older.
        Numerator: In-hospital fall-associated fractures (secondary diagnosis, not POA),
                   categorized hierarchically as Hip Fracture (priority) or Other Fracture.
        Exclusions: Principal DX of fracture, secondary DX of fracture POA='Y',
                    joint prosthesis-associated fracture, obstetric/neonatal discharges.
        """
        all_diagnoses = self._get_all_diagnoses(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Surgical or Medical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()) and drg not in appendix.get('MEDIC2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical or medical MS-DRG"

        # Exclusions (Fracture Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        # Exclusion: Principal diagnosis of fracture (FXID*)
        if principal_dx_code and principal_dx_code in appendix.get('FXID', set()):
            return "Exclusion", f"Denominator Exclusion: Principal diagnosis is fracture ({principal_dx_code})"

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: Secondary diagnosis of fracture (FXID*) present on admission (POA='Y')
            if dx_entry != principal_dx_entry and dx_code in appendix.get('FXID', set()) and poa_status == 'Y':
                return "Exclusion", f"Denominator Exclusion: Secondary fracture diagnosis ({dx_code}) present on admission (POA=Y)"

            # Exclusion: Any diagnosis of joint prosthesis-associated fracture (PROSFXID*)
            if dx_code in appendix.get('PROSFXID', set()):
                return "Exclusion", f"Denominator Exclusion: Joint prosthesis-associated fracture present ({dx_code})"

        # Numerator Identification & Hierarchy
        # Collect all non-POA secondary fractures
        non_poa_secondary_fractures: List[str] = []
        for dx_entry in all_diagnoses[1:]: # Iterate through secondary diagnoses
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']
            if dx_code in appendix.get('FXID', set()) and (poa_status in ['N', 'U', 'W', None] or pd.isna(poa_status)):
                non_poa_secondary_fractures.append(dx_code)

        if not non_poa_secondary_fractures:
            return "Exclusion", "Exclusion: No qualifying in-hospital fall-associated fracture found"

        # Apply hierarchy: Hip Fracture takes priority
        has_hip_fracture = False
        for fx_code in non_poa_secondary_fractures:
            if fx_code in appendix.get('HIPFXID', set()):
                has_hip_fracture = True
                break

        if has_hip_fracture:
            return "Inclusion", "Inclusion: In-hospital fall-associated Hip Fracture"
        else:
            # If no hip fracture, but other non-POA secondary fractures exist
            return "Inclusion", "Inclusion: In-hospital fall-associated Other Fracture"

    def evaluate_psi09(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_09: Postoperative Hemorrhage or Hematoma Rate.

        Denominator: Surgical DRG (SURGI2R), Age >=18.
        Numerator: POHMRI2D diagnosis (secondary, not POA) AND HEMOTH2P procedure.
                   HEMOTH2P must occur AFTER the first ORPROC.
        Exclusions: COAGDID (any), MEDBLEEDD (principal or secondary POA='Y'),
                    THROMBOLYTICP (before or same day as first HEMOTH2P).
        """
        # Denominator Inclusion: Surgical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical MS-DRG"

        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Get first OR procedure date for timing reference
        first_or_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ORPROC')
        if pd.isna(first_or_proc_date):
            # If no OR procedure, it cannot be "postoperative" hemorrhage in this context
            return "Exclusion", "Denominator Exclusion: No qualifying OR procedure found for timing reference"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: COAGDID (coagulation disorder) any position
            if dx_code in appendix.get('COAGDID', set()):
                return "Exclusion", f"Denominator Exclusion: Coagulation disorder diagnosis present ({dx_code})"

            # Exclusion: MEDBLEEDD (medication-related coagulopathy) if principal or secondary POA='Y'
            if dx_code in appendix.get('MEDBLEEDD', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Medication-related coagulopathy ({dx_code}) present on admission or as principal diagnosis"

        # Exclusions (Procedures) - Thrombolytic timing
        # Find the first date of a HEMOTH2P procedure to use as a reference for thrombolytic timing
        first_hemoth2p_date = self._get_first_procedure_date_by_code_set(all_procedures, 'HEMOTH2P')
        if pd.notna(first_hemoth2p_date):
            # Check if any THROMBOLYTICP occurs before or on the same day as the first HEMOTH2P
            if self._check_procedure_timing(all_procedures, first_hemoth2p_date, 'THROMBOLYTICP', max_days=0, inclusive_max=True):
                return "Exclusion", "Denominator Exclusion: Thrombolytic medication before or same day as hemorrhage treatment"

        # Numerator Check: POHMRI2D (postoperative hemorrhage/hematoma) secondary, not POA AND HEMOTH2P procedure after first ORPROC
        has_postop_hemorrhage_dx = False
        # Start from the first secondary diagnosis (DX1) onwards
        for dx_entry in all_diagnoses[1:]:
            if dx_entry['code'] in appendix.get('POHMRI2D', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_postop_hemorrhage_dx = True
                break

        if has_postop_hemorrhage_dx:
            # Check for HEMOTH2P procedure strictly after first ORPROC (min_days=0, inclusive_min=False means > 0 days after)
            if self._check_procedure_timing(all_procedures, first_or_proc_date, 'HEMOTH2P', min_days=0, inclusive_min=False):
                return "Inclusion", "Inclusion: Postoperative hemorrhage/hematoma with treatment (secondary, not POA)"

        return "Exclusion", "Exclusion: No qualifying postoperative hemorrhage/hematoma found with required treatment and timing"

    def evaluate_psi10(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_10: Postoperative Acute Kidney Injury Requiring Dialysis Rate.

        Denominator: Elective surgical discharges (ATYPE=3) for patients ages 18 years and older with OR procedures.
        Numerator: Postoperative acute kidney failure (PHYSIDB secondary, not POA) AND dialysis procedure (DIALYIP)
                   after the primary OR procedure.
        Exclusions:
            - Principal DX of PHYSIDB or Secondary DX of PHYSIDB POA='Y'.
            - DIALYIP or DIALY2P procedures before or same day as first ORPROC.
            - Principal DX of CARDIID, CARDRID, SHOCKID or Secondary DX of these POA='Y'.
            - Principal DX of CRENLFD or Secondary DX of CRENLFD POA='Y'.
            - Principal DX of URINARYOBSID.
            - SOLKIDD POA='Y' AND PNEPHREP procedure.
            - Obstetric/neonatal discharges.
        """
        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Elective Surgical Population (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical MS-DRG"

        admission_type = row.get('ATYPE')
        if pd.isna(admission_type) or int(admission_type) != 3:
            return "Exclusion", "Denominator Exclusion: Admission not elective (ATYPE != 3)"

        # Denominator Inclusion: At least one OR procedure
        first_or_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ORPROC')
        if pd.isna(first_or_proc_date):
            return "Exclusion", "Denominator Exclusion: No qualifying OR procedure found"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: Principal DX of PHYSIDB or Secondary DX of PHYSIDB POA='Y'
            if dx_code in appendix.get('PHYSIDB', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Acute kidney failure ({dx_code}) present on admission or as principal diagnosis"

            # Exclusion: Cardiac Conditions (CARDIID, CARDRID)
            if dx_code in appendix.get('CARDIID', set()) or dx_code in appendix.get('CARDRID', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Cardiac condition ({dx_code}) present on admission or as principal diagnosis"

            # Exclusion: Shock Conditions (SHOCKID)
            if dx_code in appendix.get('SHOCKID', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Shock condition ({dx_code}) present on admission or as principal diagnosis"

            # Exclusion: Chronic Kidney Disease (CRENLFD)
            if dx_code in appendix.get('CRENLFD', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Chronic kidney disease ({dx_code}) present on admission or as principal diagnosis"

            # Exclusion: Urinary Obstruction (URINARYOBSID) - only principal
            if dx_entry == principal_dx_entry and dx_code in appendix.get('URINARYOBSID', set()):
                return "Exclusion", f"Denominator Exclusion: Principal diagnosis is urinary tract obstruction ({dx_code})"

        # Exclusions (Procedures)
        # Dialysis Timing Exclusions (DIALYIP, DIALY2P before/same day as first ORPROC)
        if self._check_procedure_timing(all_procedures, first_or_proc_date, 'DIALYIP', max_days=0, inclusive_max=True):
            return "Exclusion", "Denominator Exclusion: Dialysis procedure before or same day as first OR procedure"
        if self._check_procedure_timing(all_procedures, first_or_proc_date, 'DIALY2P', max_days=0, inclusive_max=True):
            return "Exclusion", "Denominator Exclusion: Dialysis access procedure before or same day as first OR procedure"

        # Solitary Kidney Nephrectomy (SOLKIDD POA='Y' AND PNEPHREP procedure)
        has_solkid_poa = any(dx_entry['code'] in appendix.get('SOLKIDD', set()) and dx_entry['poa'] == 'Y' for dx_entry in all_diagnoses)
        has_pnephrep_proc = any(proc_entry['code'] in appendix.get('PNEPHREP', set()) for proc_entry in all_procedures)
        if has_solkid_poa and has_pnephrep_proc:
            return "Exclusion", "Denominator Exclusion: Solitary kidney present on admission with nephrectomy procedure"

        # Numerator Check: PHYSIDB (secondary, not POA) AND DIALYIP (after first ORPROC)
        has_aki_dx = False
        for dx_entry in all_diagnoses[1:]: # Iterate through secondary diagnoses
            if dx_entry['code'] in appendix.get('PHYSIDB', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_aki_dx = True
                break

        if has_aki_dx:
            # Check for DIALYIP procedure strictly after first ORPROC
            if self._check_procedure_timing(all_procedures, first_or_proc_date, 'DIALYIP', min_days=0, inclusive_min=False):
                return "Inclusion", "Inclusion: Postoperative acute kidney injury requiring dialysis"

        return "Exclusion", "Exclusion: No qualifying postoperative acute kidney injury requiring dialysis found"

    def evaluate_psi11(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_11: Postoperative Respiratory Failure Rate.

        Denominator: Elective surgical discharges (ATYPE=3) for patients ages 18 years and older with OR procedures.
        Numerator: Discharges with ANY of the following postoperative respiratory complications:
            - Acute postprocedural respiratory failure (ACURF2D secondary, not POA).
            - Mechanical ventilation > 96 consecutive hours (PR9672P) >= 0 days after first major OR procedure.
            - Mechanical ventilation 24-96 consecutive hours (PR9671P) >= 2 days after first major OR procedure.
            - Intubation procedure (PR9604P) >= 1 day after first major OR procedure.
        Exclusions:
            - Principal DX of ACURF3D or Secondary DX of ACURF3D POA='Y'.
            - Any DX of TRACHID POA='Y'.
            - Only OR procedure is TRACHIP or TRACHIP before first ORPROC.
            - Any DX of MALHYPD.
            - Any DX of NEUROMD POA='Y'.
            - Any DX of DGNEUID POA='Y'.
            - High-risk surgeries: NUCRANP, PRESOPP, LUNGCIP, LUNGTRANSP.
            - MDC 4 (Respiratory System).
            - Obstetric/neonatal discharges.
        """
        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Elective Surgical Population (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical MS-DRG"

        admission_type = row.get('ATYPE')
        if pd.isna(admission_type) or int(admission_type) != 3:
            return "Exclusion", "Denominator Exclusion: Admission not elective (ATYPE != 3)"

        # Denominator Inclusion: At least one OR procedure
        first_or_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ORPROC')
        if pd.isna(first_or_proc_date):
            return "Exclusion", "Denominator Exclusion: No qualifying OR procedure found"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: Principal DX of ACURF3D or Secondary DX of ACURF3D POA='Y'
            if dx_code in appendix.get('ACURF3D', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Acute respiratory failure ({dx_code}) present on admission or as principal diagnosis"

            # Exclusion: Any DX of TRACHID POA='Y'
            if dx_code in appendix.get('TRACHID', set()) and poa_status == 'Y':
                return "Exclusion", f"Denominator Exclusion: Tracheostomy diagnosis ({dx_code}) present on admission"

            # Exclusion: Any DX of MALHYPD
            if dx_code in appendix.get('MALHYPD', set()):
                return "Exclusion", f"Denominator Exclusion: Malignant hyperthermia diagnosis present ({dx_code})"

            # Exclusion: Any DX of NEUROMD POA='Y'
            if dx_code in appendix.get('NEUROMD', set()) and poa_status == 'Y':
                return "Exclusion", f"Denominator Exclusion: Neuromuscular disorder ({dx_code}) present on admission"

            # Exclusion: Any DX of DGNEUID POA='Y'
            if dx_code in appendix.get('DGNEUID', set()) and poa_status == 'Y':
                return "Exclusion", f"Denominator Exclusion: Degenerative neurological disorder ({dx_code}) present on admission"

        # Exclusions (Procedures)
        # Only OR procedure is tracheostomy (TRACHIP)
        or_procedures_codes = [p['code'] for p in all_procedures if p['code'] in appendix.get('ORPROC', set())]
        if len(or_procedures_codes) == 1 and or_procedures_codes[0] in appendix.get('TRACHIP', set()):
            return "Exclusion", "Denominator Exclusion: Only OR procedure is tracheostomy"

        # Tracheostomy (TRACHIP) occurs before first OR procedure
        first_trach_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'TRACHIP')
        if pd.notna(first_trach_proc_date) and pd.notna(first_or_proc_date) and first_trach_proc_date < first_or_proc_date:
            return "Exclusion", "Denominator Exclusion: Tracheostomy procedure before first OR procedure"

        # High-risk surgeries
        for proc_entry in all_procedures:
            proc_code = proc_entry['code']
            if proc_code in appendix.get('NUCRANP', set()) or \
               proc_code in appendix.get('PRESOPP', set()) or \
               proc_code in appendix.get('LUNGCIP', set()) or \
               proc_code in appendix.get('LUNGTRANSP', set()):
                return "Exclusion", f"Denominator Exclusion: High-risk surgery procedure present ({proc_code})"

        # MDC 4 (Respiratory System) Exclusion
        mdc = row.get('MDC')
        if pd.notna(mdc) and int(mdc) == 4:
            return "Exclusion", "Denominator Exclusion: MDC 4 (Diseases & Disorders of the Respiratory System)"

        # Numerator Logic (ANY of the following criteria):
        has_postop_respiratory_complication = False

        # 1. Acute postprocedural respiratory failure (ACURF2D secondary, not POA)
        for dx_entry in all_diagnoses[1:]: # Secondary diagnoses
            if dx_entry['code'] in appendix.get('ACURF2D', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_postop_respiratory_complication = True
                break

        if not has_postop_respiratory_complication:
            # 2. Mechanical ventilation > 96 consecutive hours (PR9672P) >= 0 days after first major OR procedure
            if self._check_procedure_timing(all_procedures, first_or_proc_date, 'PR9672P', min_days=0, inclusive_min=True):
                has_postop_respiratory_complication = True

        if not has_postop_respiratory_complication:
            # 3. Mechanical ventilation 24-96 consecutive hours (PR9671P) >= 2 days after first major OR procedure
            if self._check_procedure_timing(all_procedures, first_or_proc_date, 'PR9671P', min_days=2, inclusive_min=True):
                has_postop_respiratory_complication = True

        if not has_postop_respiratory_complication:
            # 4. Intubation procedure (PR9604P) >= 1 day after first major OR procedure
            if self._check_procedure_timing(all_procedures, first_or_proc_date, 'PR9604P', min_days=1, inclusive_min=True):
                has_postop_respiratory_complication = True

        if has_postop_respiratory_complication:
            return "Inclusion", "Inclusion: Postoperative respiratory failure"
        else:
            return "Exclusion", "Exclusion: No qualifying postoperative respiratory complication found"

    def evaluate_psi12(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_12: Perioperative Pulmonary Embolism or Deep Vein Thrombosis Rate.

        Denominator: Surgical DRG (SURGI2R), Age >=18, at least one OR procedure.
        Numerator: DEEPVIB or PULMOID (secondary, not POA).
        Exclusions:
            - Principal DX of DEEPVIB or PULMOID.
            - Secondary DX of DEEPVIB or PULMOID POA='Y'.
            - VENACIP or THROMP procedures before/same day as first OR.
            - First OR procedure >=10 days after admission.
            - HITD (secondary, any), NEURTRAD (any POA), ECMOP procedure (any).
        """
        # Denominator Inclusion: Surgical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical MS-DRG"

        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        admission_date, _ = self._get_admission_discharge_dates(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: At least one OR procedure
        first_or_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ORPROC')
        if pd.isna(first_or_proc_date):
            return "Exclusion", "Denominator Exclusion: No qualifying OR procedure found"

        # Exclusions (Timing-based)
        # VENACIP or THROMP procedures before/same day as first OR
        if self._check_procedure_timing(all_procedures, first_or_proc_date, 'VENACIP', max_days=0, inclusive_max=True):
            return "Exclusion", "Denominator Exclusion: Vena cava interruption before or same day as first OR procedure"
        if self._check_procedure_timing(all_procedures, first_or_proc_date, 'THROMP', max_days=0, inclusive_max=True):
            return "Exclusion", "Denominator Exclusion: Thrombectomy before or same day as first OR procedure"

        # Late surgery exclusion: first OR >=10 days after admission
        days_since_admission_to_first_or = self._calculate_days_diff(admission_date, first_or_proc_date)
        if days_since_admission_to_first_or is not None and days_since_admission_to_first_or >= 10:
            return "Exclusion", "Denominator Exclusion: First OR procedure occurred 10 or more days after admission"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: Principal DX of DEEPVIB or PULMOID
            if dx_entry == principal_dx_entry and (dx_code in appendix.get('DEEPVIB', set()) or dx_code in appendix.get('PULMOID', set())):
                return "Exclusion", f"Denominator Exclusion: Principal diagnosis is DVT/PE ({dx_code})"

            # Exclusion: Secondary DX of DEEPVIB or PULMOID POA='Y'
            # This applies to all secondary diagnoses, so check for POA='Y'
            if dx_entry != principal_dx_entry and (dx_code in appendix.get('DEEPVIB', set()) or dx_code in appendix.get('PULMOID', set())):
                 if poa_status == 'Y':
                    return "Exclusion", f"Denominator Exclusion: DVT/PE diagnosis ({dx_code}) present on admission (POA=Y)"

            # Exclusion: HITD (heparin-induced thrombocytopenia) secondary, any POA
            # The JSON states "any secondary diagnosis", not restricted by POA for exclusion.
            if dx_entry != principal_dx_entry and dx_code in appendix.get('HITD', set()):
                return "Exclusion", f"Denominator Exclusion: Heparin-induced thrombocytopenia ({dx_code}) present"

            # Exclusion: NEURTRAD (acute brain/spinal injury) any POA (but only if POA=Y for the exclusion, as per JSON)
            if dx_code in appendix.get('NEURTRAD', set()) and poa_status == 'Y':
                return "Exclusion", f"Denominator Exclusion: Acute brain or spinal injury ({dx_code}) present on admission (POA=Y)"

        # Exclusions (Procedures)
        for proc_entry in all_procedures:
            proc_code = proc_entry['code']
            # Exclusion: ECMOP (extracorporeal membrane oxygenation)
            if proc_code in appendix.get('ECMOP', set()):
                return "Exclusion", f"Denominator Exclusion: ECMO procedure present ({proc_code})"

        # Numerator Check: DEEPVIB or PULMOID (secondary, not POA)
        has_dvt_pe = False
        # Start from the first secondary diagnosis (DX1) onwards
        for dx_entry in all_diagnoses[1:]:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']
            if (dx_code in appendix.get('DEEPVIB', set()) or dx_code in appendix.get('PULMOID', set())) and \
               (poa_status in ['N', 'U', 'W', None] or pd.isna(poa_status)):
                has_dvt_pe = True
                break

        if has_dvt_pe:
            return "Inclusion", "Inclusion: Perioperative Pulmonary Embolism or Deep Vein Thrombosis (secondary, not POA)"
        else:
            return "Exclusion", "Exclusion: No qualifying perioperative DVT/PE found"

    def evaluate_psi13(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_13: Postoperative Sepsis Rate.

        Denominator: Elective surgical discharges (ATYPE=3) for patients >=18 years with OR procedures.
        Numerator: Postoperative sepsis (SEPTI2D secondary, not POA).
        Exclusions: Principal sepsis/infection, secondary sepsis/infection POA='Y',
                    first OR procedure >=10 days after admission.
        Risk Adjustment: Based on immune function severity.
        """
        # Denominator Inclusion: Surgical DRG (Age >=18 handled by base exclusions)
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical MS-DRG"

        admission_type = row.get('ATYPE')
        if pd.isna(admission_type) or int(admission_type) != 3:
            return "Exclusion", "Denominator Exclusion: Admission not elective (ATYPE != 3)"

        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        admission_date, _ = self._get_admission_discharge_dates(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"


        # Denominator Inclusion: At least one OR procedure
        first_or_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ORPROC')
        if pd.isna(first_or_proc_date):
            return "Exclusion", "Denominator Exclusion: No qualifying OR procedure found"

        # Exclusions (Timing-based)
        # First OR procedure >=10 days after admission
        days_since_admission_to_first_or = self._calculate_days_diff(admission_date, first_or_proc_date)
        if days_since_admission_to_first_or is not None and days_since_admission_to_first_or >= 10:
            return "Exclusion", "Denominator Exclusion: First OR procedure occurred 10 or more days after admission"

        # Exclusions (Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: Principal DX of SEPTI2D or INFECID
            if dx_entry == principal_dx_entry and \
               (dx_code in appendix.get('SEPTI2D', set()) or dx_code in appendix.get('INFECID', set())):
                return "Exclusion", f"Denominator Exclusion: Principal diagnosis is sepsis or infection ({dx_code})"

            # Exclusion: Secondary DX of SEPTI2D or INFECID POA='Y'
            # This applies to all secondary diagnoses, so check for POA='Y'
            if dx_entry != principal_dx_entry and \
               (dx_code in appendix.get('SEPTI2D', set()) or dx_code in appendix.get('INFECID', set())) and \
               poa_status == 'Y':
                return "Exclusion", f"Denominator Exclusion: Sepsis or infection diagnosis ({dx_code}) present on admission (POA=Y)"

        # Numerator Check: SEPTI2D (postoperative sepsis) secondary, not POA
        has_postop_sepsis = False
        # Start from the first secondary diagnosis (DX1) onwards
        for dx_entry in all_diagnoses[1:]:
            if dx_entry['code'] in appendix.get('SEPTI2D', set()) and \
               (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_postop_sepsis = True
                break

        if has_postop_sepsis:
            risk_category = self._assign_psi13_risk_category(all_diagnoses, all_procedures)
            return "Inclusion", f"Inclusion: Postoperative sepsis (secondary, not POA) - Risk Category: {risk_category}"
        else:
            return "Exclusion", "Exclusion: No qualifying postoperative sepsis found"

    def evaluate_psi14(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_14: Postoperative Wound Dehiscence Rate.

        Denominator: Discharges for patients ages 18 years and older with abdominopelvic surgery
                     (open or non-open approach).
        Numerator: Postoperative reclosure procedures involving the abdominal wall (RECLOIP)
                   AND diagnosis of disruption of internal operation (surgical) wound (ABWALLCD secondary, not POA).
                   Reclosure procedure must occur AFTER the initial abdominopelvic surgery.
        Exclusions:
            - Last RECLOIP occurs on or before first abdominopelvic surgery (open or non-open).
            - Principal DX of ABWALLCD or Secondary DX of ABWALLCD POA='Y'.
            - Length of stay less than 2 days.
            - Obstetric/neonatal discharges.
        Stratification: open_approach (priority 1) vs non_open_approach (priority 2).
        """
        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: At least one abdominopelvic procedure (ABDOMIPOPEN or ABDOMIPOTHER)
        has_abdominopelvic_proc = False
        first_abdomip_open_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ABDOMIPOPEN')
        first_abdomip_other_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ABDOMIPOTHER')

        if pd.notna(first_abdomip_open_date) or pd.notna(first_abdomip_other_date):
            has_abdominopelvic_proc = True
        
        if not has_abdominopelvic_proc:
            return "Exclusion", "Denominator Exclusion: No qualifying abdominopelvic procedure found"

        # Determine the earliest abdominopelvic procedure date for timing comparisons
        initial_abdomip_proc_date = pd.NaT
        if pd.notna(first_abdomip_open_date) and pd.notna(first_abdomip_other_date):
            initial_abdomip_proc_date = min(first_abdomip_open_date, first_abdomip_other_date)
        elif pd.notna(first_abdomip_open_date):
            initial_abdomip_proc_date = first_abdomip_open_date
        elif pd.notna(first_abdomip_other_date):
            initial_abdomip_proc_date = first_abdomip_other_date

        if pd.isna(initial_abdomip_proc_date):
            return "Exclusion", "Data Exclusion: Missing date for initial abdominopelvic procedure"

        # Denominator Inclusion: Length of Stay >= 2 days
        los = row.get('Length_of_stay')
        if pd.isna(los) or int(los) < 2:
            return "Exclusion", "Denominator Exclusion: Length of stay less than 2 days or missing"

        # Exclusions (Procedure Timing)
        last_recloip_date = self._get_latest_procedure_date_by_code_set(all_procedures, 'RECLOIP')

        # If RECLOIP exists, check its timing relative to the initial abdominopelvic procedure
        if pd.notna(last_recloip_date) and last_recloip_date <= initial_abdomip_proc_date:
            return "Exclusion", "Denominator Exclusion: Reclosure procedure occurred on or before initial abdominopelvic surgery"

        # Exclusions (Clinical Condition - Diagnoses)
        principal_dx_entry = all_diagnoses[0]
        principal_dx_code = principal_dx_entry['code']

        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            poa_status = dx_entry['poa']

            # Exclusion: Principal DX of ABWALLCD or Secondary DX of ABWALLCD POA='Y'
            if dx_code in appendix.get('ABWALLCD', set()):
                if (dx_entry == principal_dx_entry) or (poa_status == 'Y'):
                    return "Exclusion", f"Denominator Exclusion: Wound dehiscence diagnosis ({dx_code}) present on admission or as principal diagnosis"

        # Numerator Logic: RECLOIP procedure AND ABWALLCD diagnosis (secondary, not POA)
        has_recloip_proc = any(p['code'] in appendix.get('RECLOIP', set()) for p in all_procedures)

        has_abwallcd_dx = False
        for dx_entry in all_diagnoses[1:]: # Secondary diagnoses
            if dx_entry['code'] in appendix.get('ABWALLCD', set()) and (dx_entry['poa'] in ['N', 'U', 'W', None] or pd.isna(dx_entry['poa'])):
                has_abwallcd_dx = True
                break

        if has_recloip_proc and has_abwallcd_dx:
            # Final check for numerator: RECLOIP must be *after* initial abdominopelvic surgery
            # This is already covered by the exclusion check `last_recloip_date <= initial_abdomip_proc_date`
            # If we reached here, it means last_recloip_date > initial_abdomip_proc_date (or last_recloip_date is NaT, but then has_recloip_proc would be false)
            
            stratum = self._assign_psi14_stratum(all_procedures)
            return "Inclusion", f"Inclusion: Postoperative wound dehiscence - Stratum: {stratum}"
        else:
            return "Exclusion", "Exclusion: No qualifying postoperative wound dehiscence found"

    def evaluate_psi15(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_15: Abdominopelvic Accidental Puncture or Laceration Rate.

        Denominator: Medical or Surgical DRG (MEDIC2R/SURGI2R), Age >=18, and at least one
                     Abdominopelvic procedure (ABDOMI15P).
        Numerator (Triple Criteria):
            1. Organ-specific injury diagnosis (secondary, not POA)
            2. Related procedure for SAME organ system
            3. Related procedure occurs 1-30 days after the first ABDOMI15P (index procedure)

        Exclusions:
            - Principal DX of accidental puncture/laceration.
            - Secondary DX of accidental puncture/laceration POA='Y' AND a matching related procedure for the same organ.
            - Missing index abdominopelvic procedure date.
        """
        # Denominator Inclusion: Medical or Surgical DRG
        drg = str(row.get('MS-DRG'))
        if drg not in appendix.get('SURGI2R', set()) and drg not in appendix.get('MEDIC2R', set()):
            return "Exclusion", "Denominator Exclusion: Not a surgical or medical MS-DRG"

        all_procedures = self._get_all_procedures(row)
        all_diagnoses = self._get_all_diagnoses(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: At least one abdominopelvic procedure (ABDOMI15P)
        first_abdomi_proc_date = self._get_first_procedure_date_by_code_set(all_procedures, 'ABDOMI15P')
        if pd.isna(first_abdomi_proc_date):
            return "Exclusion", "Denominator Exclusion: No qualifying abdominopelvic procedure (ABDOMI15P) or missing date"

        # Exclusion: Principal diagnosis of accidental puncture/laceration
        # Check if Pdx (principal diagnosis) is in any of the PSI_15 injury diagnosis code sets
        principal_dx_code = all_diagnoses[0]['code']
        if principal_dx_code:
            # Check against the consolidated set of all PSI_15 injury DX codes
            if principal_dx_code in self.all_psi15_injury_dx_codes:
                return "Exclusion", f"Denominator Exclusion: Principal diagnosis is accidental puncture/laceration ({principal_dx_code})"

        # Numerator Logic: Find a triple-criteria match
        has_qualifying_case = False
        # Iterate through secondary diagnoses only for numerator check
        for injury_dx_entry in all_diagnoses[1:]: # Start from DX1 onwards
            injury_dx_code = injury_dx_entry['code']
            injury_poa_status = injury_dx_entry['poa']

            # Get the organ system for the current injury diagnosis
            injury_organ_system = self._get_organ_system_from_code(injury_dx_code, is_dx=True)

            if not injury_organ_system: # Not a recognized PSI_15 injury diagnosis
                continue

            # Check for POA exclusion: Secondary DX POA='Y' AND a matching related procedure for the same organ.
            if injury_poa_status == 'Y':
                matching_proc_code_set_name = self.organ_system_mappings[injury_organ_system]['proc_codes']
                # Only exclude if a matching procedure is found within the specified time window
                # We need to check if ANY procedure from the matching_proc_code_set_name exists
                # within the 1-30 day window relative to the first_abdomi_proc_date.
                # If such a procedure exists AND the injury is POA='Y', then exclude.
                if self._check_procedure_timing(all_procedures, first_abdomi_proc_date, matching_proc_code_set_name, min_days=1, max_days=30, inclusive_min=True, inclusive_max=True):
                     return "Exclusion", f"Denominator Exclusion: POA accidental puncture/laceration ({injury_dx_code}) with matching related procedure"
                else:
                    # If POA='Y' but no matching procedure in window, it's not an exclusion, continue to check numerator.
                    pass # Do not return, continue processing this diagnosis for numerator if it's not excluded by POA rule

            # If not excluded by POA, check for numerator inclusion
            # Find a related procedure for the same organ system within the 1-30 day window
            matching_proc_code_set_name = self.organ_system_mappings[injury_organ_system]['proc_codes']
            if self._check_procedure_timing(all_procedures, first_abdomi_proc_date, matching_proc_code_set_name, min_days=1, max_days=30, inclusive_min=True, inclusive_max=True):
                has_qualifying_case = True
                break # Found a qualifying case for numerator

        if has_qualifying_case:
            risk_category = self._assign_psi15_risk_category(all_procedures, first_abdomi_proc_date)
            return "Inclusion", f"Inclusion: Abdominopelvic accidental puncture/laceration - Risk Category: {risk_category}"
        else:
            return "Exclusion", "Exclusion: No qualifying abdominopelvic accidental puncture/laceration found"

    def evaluate_psi17(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_17: Birth Trauma Rate – Injury to Neonate.

        Denominator: All newborn discharges (NEWBORN codes).
        Numerator: BIRTHID codes (any position).
        Exclusions: PRETEID (<2000g) (any), OSTEOID (any), MDC14PRINDX (principal).
        """
        all_diagnoses = self._get_all_diagnoses(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Newborn Population (handled by base exclusions for population_type 'newborn_only')
        # Additional check for NEWBORN codes if not already covered by MDC15PRINDX in base exclusions
        # The JSON indicates `Appendix M: NEWBORN` for the denominator, which is a set of DX codes.
        # Ensure the principal diagnosis is in the NEWBORN set.
        principal_dx_code = all_diagnoses[0]['code'] if all_diagnoses else None
        if not (principal_dx_code and principal_dx_code in appendix.get('NEWBORN', set())):
             return "Exclusion", "Denominator Exclusion: Not a newborn discharge (Principal DX not in NEWBORN codes)"


        # Exclusions (Clinical)
        for dx_entry in all_diagnoses:
            dx_code = dx_entry['code']
            # Exclusion: PRETEID (preterm infant <2000g) any position
            if dx_code in appendix.get('PRETEID', set()):
                return "Exclusion", f"Denominator Exclusion: Preterm infant with birth weight < 2000g ({dx_code})"
            # Exclusion: OSTEOID (osteogenesis imperfecta) any position
            if dx_code in appendix.get('OSTEOID', set()):
                return "Exclusion", f"Denominator Exclusion: Osteogenesis imperfecta diagnosis present ({dx_code})"

        # Numerator Check: BIRTHID (birth trauma injury) any position
        has_birth_trauma = False
        for dx_entry in all_diagnoses:
            if dx_entry['code'] in appendix.get('BIRTHID', set()):
                has_birth_trauma = True
                break

        if has_birth_trauma:
            return "Inclusion", "Inclusion: Birth trauma injury to neonate"
        else:
            return "Exclusion", "Exclusion: No qualifying birth trauma injury found"

    def evaluate_psi18(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_18: Obstetric Trauma Rate – Vaginal Delivery With Instrument.

        Denominator: Instrument-assisted vaginal deliveries (DELOCMD, VAGDELP, INSTRIP).
        Numerator: OBTRAID codes (any position).
        Exclusions: MDC15PRINDX (principal).
        """
        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Delivery Outcome Diagnosis (DELOCMD)
        has_delivery_outcome_dx = any(dx_entry['code'] in appendix.get('DELOCMD', set()) for dx_entry in all_diagnoses)
        if not has_delivery_outcome_dx:
            return "Exclusion", "Denominator Exclusion: No delivery outcome diagnosis found"

        # Denominator Inclusion: Vaginal Delivery Procedure (VAGDELP)
        has_vaginal_delivery_proc = any(proc_entry['code'] in appendix.get('VAGDELP', set()) for proc_entry in all_procedures)
        if not has_vaginal_delivery_proc:
            return "Exclusion", "Denominator Exclusion: No vaginal delivery procedure found"

        # Denominator Inclusion: Instrument-Assisted Delivery Procedure (INSTRIP)
        has_instrument_assisted_proc = any(proc_entry['code'] in appendix.get('INSTRIP', set()) for proc_entry in all_procedures)
        if not has_instrument_assisted_proc:
            return "Exclusion", "Denominator Exclusion: No instrument-assisted delivery procedure found"

        # Numerator Check: OBTRAID (third or fourth degree obstetric injury) any position
        has_obstetric_trauma = False
        for dx_entry in all_diagnoses:
            if dx_entry['code'] in appendix.get('OBTRAID', set()):
                has_obstetric_trauma = True
                break

        if has_obstetric_trauma:
            return "Inclusion", "Inclusion: Obstetric trauma (third or fourth degree) with instrument-assisted vaginal delivery"
        else:
            return "Exclusion", "Exclusion: No qualifying obstetric trauma found for instrument-assisted vaginal delivery"

    def evaluate_psi19(self, row: pd.Series, appendix: Dict[str, Set[str]]) -> Tuple[str, str]:
        """
        Evaluates a patient encounter for PSI_19: Obstetric Trauma Rate – Vaginal Delivery Without Instrument.

        Denominator: Spontaneous vaginal deliveries (DELOCMD, VAGDELP, NOT INSTRIP).
        Numerator: OBTRAID codes (any position).
        Exclusions: INSTRIP (any), MDC15PRINDX (principal).
        """
        all_diagnoses = self._get_all_diagnoses(row)
        all_procedures = self._get_all_procedures(row)
        if not all_diagnoses:
            return "Exclusion", "Data Exclusion: No diagnoses found"

        # Denominator Inclusion: Delivery Outcome Diagnosis (DELOCMD)
        has_delivery_outcome_dx = any(dx_entry['code'] in appendix.get('DELOCMD', set()) for dx_entry in all_diagnoses)
        if not has_delivery_outcome_dx:
            return "Exclusion", "Denominator Exclusion: No delivery outcome diagnosis found"

        # Denominator Inclusion: Vaginal Delivery Procedure (VAGDELP)
        has_vaginal_delivery_proc = any(proc_entry['code'] in appendix.get('VAGDELP', set()) for proc_entry in all_procedures)
        if not has_vaginal_delivery_proc:
            return "Exclusion", "Denominator Exclusion: No vaginal delivery procedure found"

        # Denominator Exclusion: Instrument-Assisted Delivery Procedure (INSTRIP)
        has_instrument_assisted_proc = any(proc_entry['code'] in appendix.get('INSTRIP', set()) for proc_entry in all_procedures)
        if has_instrument_assisted_proc:
            return "Exclusion", "Denominator Exclusion: Instrument-assisted delivery procedure found (PSI_19 excludes these)"

        # Numerator Check: OBTRAID (third or fourth degree obstetric injury) any position
        has_obstetric_trauma = False
        for dx_entry in all_diagnoses:
            if dx_entry['code'] in appendix.get('OBTRAID', set()):
                has_obstetric_trauma = True
                break

        if has_obstetric_trauma:
            return "Inclusion", "Inclusion: Obstetric trauma (third or fourth degree) with spontaneous vaginal delivery"
        else:
            return "Exclusion", "Exclusion: No qualifying obstetric trauma found for spontaneous vaginal delivery"

if __name__ == "__main__":
    import pandas as pd
    import json

    # Load input Excel
    input_excel = "PSI_Master_Input_Template_With_Disposition.xlsx"
    df = pd.read_excel(input_excel)

    # Load appendix code sets
    with open("PSI_02_19_Compiled_Stitched_Final.json", "r") as f:
        appendix = json.load(f)

    # Initialize calculator
    calculator = PSICalculator()

    # Prepare output
    output_rows = []

    # Loop through each encounter and evaluate all PSIs
    for idx, row in df.iterrows():
        encounter_id = row.get("EncounterID", f"Row{idx+1}")
        for psi_number in range(2, 20):
            psi_code = f"PSI_{psi_number:02}"
            eval_func = getattr(calculator, f"evaluate_psi{psi_number:02}", None)
            if eval_func:
                status, rationale = eval_func(row, appendix)
                output_rows.append({
                    "EncounterID": encounter_id,
                    "PSI": psi_code,
                    "Status": status,
                    "Rationale": rationale
                })

    # Export result
    result_df = pd.DataFrame(output_rows)
    result_df.to_excel("PSI_02_19_Output_Result.xlsx", index=False)
    print("✅ Analysis complete. Output saved to PSI_02_19_Output_Result.xlsx")
