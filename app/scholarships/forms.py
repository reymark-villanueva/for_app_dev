from django import forms

from .models import StudentProfile


class StudentProfileForm(forms.Form):
    # -- Personal Information --
    first_name = forms.CharField(max_length=100)
    last_name = forms.CharField(max_length=100)
    age = forms.IntegerField(min_value=15, max_value=65)
    sex = forms.ChoiceField(choices=StudentProfile.SEX_CHOICES)
    civil_status = forms.ChoiceField(choices=StudentProfile.CIVIL_STATUS_CHOICES)

    # -- Academic Information --
    year_level = forms.TypedChoiceField(
        choices=StudentProfile.YEAR_LEVEL_CHOICES,
        coerce=int,
        label='Year Level',
    )
    gwa = forms.FloatField(
        min_value=1.0, max_value=5.0,
        label='GWA (1.0 - 5.0)',
        help_text='General Weighted Average on a 1.0 to 5.0 scale',
    )
    course_combined = forms.ChoiceField(
        choices=[('', '-- Select Course --')] + StudentProfile.COURSE_CHOICES,
        label='Course',
    )

    # -- Financial --
    income_range = forms.ChoiceField(
        choices=[('', '-- Select Income Range --')] + StudentProfile.INCOME_RANGE_CHOICES,
        label='Family Annual Income',
    )
    parents_occupation = forms.CharField(max_length=200)

    # -- Vulnerability Flags --
    is_solo_parent_dependent = forms.BooleanField(required=False, label='Solo Parent Dependent')
    is_pwd = forms.BooleanField(required=False, label='Person with Disability (PWD)')
    is_indigenous_people = forms.BooleanField(required=False, label='Indigenous People')
    is_4ps_beneficiary = forms.BooleanField(required=False, label='4Ps Beneficiary')
    is_ofw_dependent = forms.BooleanField(required=False, label='OFW Dependent')
    has_existing_scholarship = forms.BooleanField(required=False, label='Has Existing Scholarship')

    def clean_course_combined(self):
        value = self.cleaned_data['course_combined']
        if '|||' not in value:
            raise forms.ValidationError('Please select a valid course.')
        return value

    def clean(self):
        cleaned_data = super().clean()

        # Split course_combined into course_category and course
        course_combined = cleaned_data.get('course_combined', '')
        if '|||' in course_combined:
            category, course_name = course_combined.split('|||', 1)
            cleaned_data['course_category'] = category
            cleaned_data['course'] = course_name
        else:
            cleaned_data['course_category'] = ''
            cleaned_data['course'] = ''

        # Derive GWA values from the single 1-5 scale input
        gwa = cleaned_data.get('gwa')
        if gwa is not None:
            cleaned_data['gwa_numeric_1to5'] = gwa
            cleaned_data['gwa_percentage'] = 100 - (gwa - 1) * 7.5

        # Map income range to midpoint
        income_key = cleaned_data.get('income_range', '')
        cleaned_data['family_annual_income_php'] = (
            StudentProfile.INCOME_MIDPOINTS.get(income_key, 15000)
        )

        return cleaned_data

    def to_student_dict(self):
        """Convert cleaned form data to the dict format expected by the ML engine."""
        cd = self.cleaned_data
        return {
            'first_name': cd['first_name'],
            'last_name': cd['last_name'],
            'age': cd['age'],
            'sex': cd['sex'],
            'civil_status': cd['civil_status'],
            'year_level': cd['year_level'],
            'gwa_percentage': cd['gwa_percentage'],
            'gwa_numeric_1to5': cd['gwa_numeric_1to5'],
            'course': cd['course'],
            'course_category': cd['course_category'],
            # Defaults for removed fields (ML preprocessor expects them)
            'shs_strand': 'STEM',
            'enrolled_hei_type': 'SUC',
            'region': 'Region II',
            'barangay_type': 'Rural',
            'family_annual_income_php': cd['family_annual_income_php'],
            'family_size': 4,
            'parents_occupation': cd['parents_occupation'],
            'is_solo_parent_dependent': 'Yes' if cd['is_solo_parent_dependent'] else 'No',
            'is_pwd': 'Yes' if cd['is_pwd'] else 'No',
            'is_indigenous_people': 'Yes' if cd['is_indigenous_people'] else 'No',
            'is_4ps_beneficiary': 'Yes' if cd['is_4ps_beneficiary'] else 'No',
            'is_ofw_dependent': 'Yes' if cd['is_ofw_dependent'] else 'No',
            'has_existing_scholarship': 'Yes' if cd['has_existing_scholarship'] else 'No',
        }
