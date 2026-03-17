from django.conf import settings
from django.db import models


class Scholarship(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, default='')
    eligibility_notes = models.TextField(blank=True, default='')
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class StudentProfile(models.Model):
    SEX_CHOICES = [('Male', 'Male'), ('Female', 'Female')]
    CIVIL_STATUS_CHOICES = [
        ('Single', 'Single'), ('Married', 'Married'), ('Widowed', 'Widowed'),
    ]

    YEAR_LEVEL_CHOICES = [
        (1, '1st Year'),
        (2, '2nd Year'),
        (3, '3rd Year'),
        (4, '4th Year'),
        (5, '5th Year'),
    ]

    COURSE_CHOICES = [
        ('Agriculture', (
            ('Agriculture|||BS Agriculture', 'BS Agriculture'),
            ('Agriculture|||BS Agricultural Engineering', 'BS Agricultural Engineering'),
            ('Agriculture|||BS Fisheries', 'BS Fisheries'),
        )),
        ('Business', (
            ('Business|||BS Accountancy', 'BS Accountancy'),
            ('Business|||BS Business Administration', 'BS Business Administration'),
            ('Business|||BS Entrepreneurship', 'BS Entrepreneurship'),
        )),
        ('Coconut-Related', (
            ('Coconut-Related|||BS Agricultural Engineering (Coconut)', 'BS Agricultural Engineering (Coconut)'),
            ('Coconut-Related|||BS Food Technology (Coconut)', 'BS Food Technology (Coconut)'),
        )),
        ('Education', (
            ('Education|||Bachelor of Elementary Education', 'Bachelor of Elementary Education'),
            ('Education|||Bachelor of Secondary Education', 'Bachelor of Secondary Education'),
            ('Education|||BS Education', 'BS Education'),
        )),
        ('Engineering', (
            ('Engineering|||BS Civil Engineering', 'BS Civil Engineering'),
            ('Engineering|||BS Electrical Engineering', 'BS Electrical Engineering'),
            ('Engineering|||BS Mechanical Engineering', 'BS Mechanical Engineering'),
            ('Engineering|||BS Computer Engineering', 'BS Computer Engineering'),
            ('Engineering|||BS Chemical Engineering', 'BS Chemical Engineering'),
        )),
        ('Health Sciences', (
            ('Health Sciences|||BS Nursing', 'BS Nursing'),
            ('Health Sciences|||BS Pharmacy', 'BS Pharmacy'),
            ('Health Sciences|||BS Medical Technology', 'BS Medical Technology'),
            ('Health Sciences|||BS Physical Therapy', 'BS Physical Therapy'),
        )),
        ('Hospitality', (
            ('Hospitality|||BS Hospitality Management', 'BS Hospitality Management'),
            ('Hospitality|||BS Tourism Management', 'BS Tourism Management'),
        )),
        ('IT/Computing', (
            ('IT/Computing|||BS Information Technology', 'BS Information Technology'),
            ('IT/Computing|||BS Computer Science', 'BS Computer Science'),
            ('IT/Computing|||BS Information Systems', 'BS Information Systems'),
        )),
        ('Science', (
            ('Science|||BS Biology', 'BS Biology'),
            ('Science|||BS Environmental Science', 'BS Environmental Science'),
            ('Science|||BS Chemistry', 'BS Chemistry'),
            ('Science|||BS Mathematics', 'BS Mathematics'),
        )),
        ('Social Sciences', (
            ('Social Sciences|||BS Social Work', 'BS Social Work'),
            ('Social Sciences|||BS Psychology', 'BS Psychology'),
            ('Social Sciences|||BA Political Science', 'BA Political Science'),
        )),
        ('Sugarcane-Related', (
            ('Sugarcane-Related|||BS Chemical Engineering (Sugar Processing)', 'BS Chemical Engineering (Sugar Processing)'),
            ('Sugarcane-Related|||BS Agriculture (Sugar Technology)', 'BS Agriculture (Sugar Technology)'),
        )),
    ]

    INCOME_RANGE_CHOICES = [
        ('10000-19999', '\u20B110,000 - \u20B119,999'),
        ('20000-29999', '\u20B120,000 - \u20B129,999'),
        ('30000-39999', '\u20B130,000 - \u20B139,999'),
        ('40000-49999', '\u20B140,000 - \u20B149,999'),
        ('50000-59999', '\u20B150,000 - \u20B159,999'),
        ('60000-69999', '\u20B160,000 - \u20B169,999'),
        ('70000-79999', '\u20B170,000 - \u20B179,999'),
        ('80000-89999', '\u20B180,000 - \u20B189,999'),
        ('90000-100000', '\u20B190,000 - \u20B1100,000'),
    ]

    INCOME_MIDPOINTS = {
        '10000-19999': 15000,
        '20000-29999': 25000,
        '30000-39999': 35000,
        '40000-49999': 45000,
        '50000-59999': 55000,
        '60000-69999': 65000,
        '70000-79999': 75000,
        '80000-89999': 85000,
        '90000-100000': 95000,
    }

    # Keep old choices for reference / admin
    COURSE_CATEGORY_CHOICES = [
        ('Agriculture', 'Agriculture'),
        ('Business', 'Business'),
        ('Coconut-Related', 'Coconut-Related'),
        ('Education', 'Education'),
        ('Engineering', 'Engineering'),
        ('Health Sciences', 'Health Sciences'),
        ('Hospitality', 'Hospitality'),
        ('IT/Computing', 'IT/Computing'),
        ('Science', 'Science'),
        ('Social Sciences', 'Social Sciences'),
        ('Sugarcane-Related', 'Sugarcane-Related'),
        ('Other', 'Other'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='student_profiles',
    )

    # Personal
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    age = models.PositiveIntegerField()
    sex = models.CharField(max_length=10, choices=SEX_CHOICES)
    civil_status = models.CharField(max_length=20, choices=CIVIL_STATUS_CHOICES)

    # Academic
    year_level = models.PositiveIntegerField(choices=YEAR_LEVEL_CHOICES)
    gwa_percentage = models.FloatField()
    gwa_numeric_1to5 = models.FloatField()
    course = models.CharField(max_length=200)
    course_category = models.CharField(max_length=50, choices=COURSE_CATEGORY_CHOICES)

    # Removed from form but kept nullable for ML defaults
    shs_strand = models.CharField(max_length=20, blank=True, default='STEM')
    enrolled_hei_type = models.CharField(max_length=20, blank=True, default='SUC')
    region = models.CharField(max_length=50, blank=True, default='Region II')
    barangay_type = models.CharField(max_length=10, blank=True, default='Rural')
    family_size = models.PositiveIntegerField(default=4)

    # Financial
    income_range = models.CharField(max_length=20, choices=INCOME_RANGE_CHOICES, default='10000-19999')
    family_annual_income_php = models.FloatField()
    parents_occupation = models.CharField(max_length=200)

    # Vulnerability flags
    is_solo_parent_dependent = models.BooleanField(default=False)
    is_pwd = models.BooleanField(default=False)
    is_indigenous_people = models.BooleanField(default=False)
    is_4ps_beneficiary = models.BooleanField(default=False)
    is_ofw_dependent = models.BooleanField(default=False)
    has_existing_scholarship = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.first_name} {self.last_name}"


class Recommendation(models.Model):
    student_profile = models.ForeignKey(
        StudentProfile, on_delete=models.CASCADE, related_name='recommendations',
    )
    scholarship = models.ForeignKey(
        Scholarship, on_delete=models.CASCADE, related_name='recommendations',
    )
    rank = models.PositiveIntegerField()
    confidence_score = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['student_profile', 'rank']
        unique_together = ['student_profile', 'rank']

    def __str__(self):
        return f"#{self.rank} {self.scholarship.name} ({self.confidence_score:.1f}%)"
