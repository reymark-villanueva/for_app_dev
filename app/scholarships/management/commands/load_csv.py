import pandas as pd
from django.core.management.base import BaseCommand

from scholarships.models import Scholarship, StudentProfile
from scholarships.ml.cleaning import (
    clean_sex, clean_civil, clean_hei, clean_strand,
    clean_bar, clean_cat, clean_region, clean_binary,
)


SCHOLARSHIP_LABELS = [
    'ACEF-GIAHEP',
    'BRO-ED ISU Cauayan',
    'CHED CoScho',
    'CHED Merit - Full',
    'CHED Merit - Half',
    'CHED SIDA',
    'CHED SIKAP',
    'CHED TDP',
    'CHED TES',
    'College Scholar',
    'DOST Undergraduate Scholarship',
    'No Scholarship Recommended',
    'University Scholar',
]


class Command(BaseCommand):
    help = 'Load scholarship dataset CSV into the database'

    def add_arguments(self, parser):
        parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
        parser.add_argument(
            '--clear', action='store_true',
            help='Delete existing data before loading',
        )

    def handle(self, *args, **options):
        csv_path = options['csv']

        if options['clear']:
            StudentProfile.objects.all().delete()
            Scholarship.objects.all().delete()
            self.stdout.write('Cleared existing data.')

        # Create scholarship records
        for label in SCHOLARSHIP_LABELS:
            Scholarship.objects.get_or_create(name=label)
        self.stdout.write(f'Ensured {len(SCHOLARSHIP_LABELS)} scholarship records exist.')

        # Load CSV
        df = pd.read_csv(csv_path)
        self.stdout.write(f'Loaded {len(df)} rows from {csv_path}')

        created = 0
        for _, row in df.iterrows():
            try:
                gwa_pct = pd.to_numeric(row.get('gwa_percentage'), errors='coerce')
                gwa_num = pd.to_numeric(row.get('gwa_numeric_1to5'), errors='coerce')
                income = pd.to_numeric(row.get('family_annual_income_php'), errors='coerce')
                fam_size = pd.to_numeric(row.get('family_size'), errors='coerce')
                age = pd.to_numeric(row.get('age'), errors='coerce')
                year_level = pd.to_numeric(row.get('year_level'), errors='coerce')

                # Cross-fill GWA
                if pd.isna(gwa_pct) and not pd.isna(gwa_num):
                    gwa_pct = 100 - (gwa_num - 1) * 7.5
                if pd.isna(gwa_num) and not pd.isna(gwa_pct):
                    gwa_num = (100 - gwa_pct) / 7.5 + 1

                def _bool(v):
                    val = clean_binary(v)
                    return val == 1.0 if not pd.isna(val) else False

                StudentProfile.objects.create(
                    first_name=str(row.get('first_name', '')).strip(),
                    last_name=str(row.get('last_name', '')).strip(),
                    age=int(age) if not pd.isna(age) else 18,
                    sex=clean_sex(row.get('sex')),
                    civil_status=clean_civil(row.get('civil_status')),
                    year_level=int(year_level) if not pd.isna(year_level) else 1,
                    gwa_percentage=float(gwa_pct) if not pd.isna(gwa_pct) else 0.0,
                    gwa_numeric_1to5=float(gwa_num) if not pd.isna(gwa_num) else 0.0,
                    course=str(row.get('course', '')).strip(),
                    course_category=clean_cat(row.get('course_category')),
                    shs_strand=clean_strand(row.get('shs_strand')),
                    enrolled_hei_type=clean_hei(row.get('enrolled_hei_type')),
                    region=clean_region(row.get('region')),
                    barangay_type=clean_bar(row.get('barangay_type')),
                    family_annual_income_php=float(income) if not pd.isna(income) else 0.0,
                    family_size=int(fam_size) if not pd.isna(fam_size) else 1,
                    parents_occupation=str(row.get('parents_occupation', '')).strip(),
                    is_solo_parent_dependent=_bool(row.get('is_solo_parent_dependent')),
                    is_pwd=_bool(row.get('is_pwd')),
                    is_indigenous_people=_bool(row.get('is_indigenous_people')),
                    is_4ps_beneficiary=_bool(row.get('is_4ps_beneficiary')),
                    is_ofw_dependent=_bool(row.get('is_ofw_dependent')),
                    has_existing_scholarship=_bool(row.get('has_existing_scholarship')),
                )
                created += 1
            except Exception as e:
                self.stderr.write(f'Row {_} skipped: {e}')

        self.stdout.write(self.style.SUCCESS(f'Created {created} student profiles.'))
