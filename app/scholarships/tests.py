import csv
import os
import tempfile
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.test import TestCase
from django.urls import reverse

from .models import StudentProfile


class StudentFormViewTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username='tester',
            password='password123',
        )
        self.client.force_login(self.user)

    def test_student_form_rolls_back_when_models_are_missing(self):
        data = {
            'first_name': 'Jane',
            'last_name': 'Doe',
            'age': '20',
            'sex': 'Female',
            'civil_status': 'Single',
            'year_level': '1',
            'gwa': '2.0',
            'course_combined': 'Engineering|||BS Civil Engineering',
            'income_range': '10000-19999',
            'parents_occupation': 'Farmer',
        }

        with patch('scholarships.views.recommend_scholarship', side_effect=FileNotFoundError) as mock_recommend:
            response = self.client.post(reverse('scholarships:student_form'), data=data)

        self.assertEqual(response.status_code, 200)
        mock_recommend.assert_called_once()
        self.assertEqual(StudentProfile.objects.count(), 0)
        self.assertIn('form', response.context)


class LoadCsvCommandTests(TestCase):
    def test_load_csv_reports_row_number_when_row_creation_fails(self):
        fd, csv_path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)

        fieldnames = [
            'first_name', 'last_name', 'age', 'sex', 'civil_status', 'year_level',
            'gwa_percentage', 'gwa_numeric_1to5', 'course', 'course_category',
            'shs_strand', 'enrolled_hei_type', 'region', 'barangay_type',
            'family_annual_income_php', 'family_size', 'parents_occupation',
            'is_solo_parent_dependent', 'is_pwd', 'is_indigenous_people',
            'is_4ps_beneficiary', 'is_ofw_dependent', 'has_existing_scholarship',
        ]

        rows = [
            {
                'first_name': 'Ana',
                'last_name': 'Cruz',
                'age': '19',
                'sex': 'Female',
                'civil_status': 'Single',
                'year_level': '1',
                'gwa_percentage': '95',
                'gwa_numeric_1to5': '1.5',
                'course': 'BS Civil Engineering',
                'course_category': 'Engineering',
                'shs_strand': 'STEM',
                'enrolled_hei_type': 'SUC',
                'region': 'Region II',
                'barangay_type': 'Rural',
                'family_annual_income_php': '15000',
                'family_size': '4',
                'parents_occupation': 'Farmer',
                'is_solo_parent_dependent': 'yes',
                'is_pwd': 'no',
                'is_indigenous_people': 'no',
                'is_4ps_beneficiary': 'no',
                'is_ofw_dependent': 'no',
                'has_existing_scholarship': 'no',
            },
            {
                'first_name': 'Ben',
                'last_name': 'Reyes',
                'age': '20',
                'sex': 'Male',
                'civil_status': 'Single',
                'year_level': '2',
                'gwa_percentage': '90',
                'gwa_numeric_1to5': '2.0',
                'course': 'BS Computer Science',
                'course_category': 'IT/Computing',
                'shs_strand': 'STEM',
                'enrolled_hei_type': 'SUC',
                'region': 'Region II',
                'barangay_type': 'Urban',
                'family_annual_income_php': '25000',
                'family_size': '5',
                'parents_occupation': 'Driver',
                'is_solo_parent_dependent': 'no',
                'is_pwd': 'no',
                'is_indigenous_people': 'no',
                'is_4ps_beneficiary': 'no',
                'is_ofw_dependent': 'no',
                'has_existing_scholarship': 'no',
            },
        ]

        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        original_create = StudentProfile.objects.create
        call_count = {'value': 0}

        def create_side_effect(*args, **kwargs):
            call_count['value'] += 1
            if call_count['value'] == 2:
                raise ValueError('boom')
            return original_create(*args, **kwargs)

        try:
            with patch('scholarships.management.commands.load_csv.StudentProfile.objects.create', side_effect=create_side_effect):
                stdout = tempfile.SpooledTemporaryFile(mode='w+', encoding='utf-8')
                stderr = tempfile.SpooledTemporaryFile(mode='w+', encoding='utf-8')
                call_command('load_csv', csv=csv_path, stdout=stdout, stderr=stderr)
                stdout.seek(0)
                stderr.seek(0)
                output = stdout.read()
                error_output = stderr.read()
        finally:
            os.remove(csv_path)

        self.assertIn('Created 1 student profiles.', output)
        self.assertIn('Row 2 skipped: boom', error_output)
        self.assertEqual(StudentProfile.objects.count(), 1)
