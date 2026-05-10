import logging

from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db import transaction
from django.shortcuts import render, redirect, get_object_or_404

from .forms import StudentProfileForm
from .models import StudentProfile, Scholarship, Recommendation
from .ml.engine import recommend_scholarship


logger = logging.getLogger(__name__)


# Scholarship eligibility rules based on the ML notebook
SCHOLARSHIP_RULES = {
    'University Scholar': {
        'gwa_min': 97.0,
        'gwa_reason': 'Outstanding GWA of 97%+ qualifies you',
        'category': None,
    },
    'College Scholar': {
        'gwa_min': 95.0,
        'gwa_max': 97.0,
        'gwa_reason': 'GWA between 95-97% qualifies you',
        'category': None,
    },
    'CHED Merit - Full': {
        'gwa_min': 96.0,
        'gwa_max': 99.5,
        'gwa_reason': 'GWA qualifies for full merit scholarship',
        'category': None,
    },
    'CHED Merit - Half': {
        'gwa_min': 93.0,
        'gwa_max': 95.0,
        'gwa_reason': 'GWA between 93-95% qualifies for half merit',
        'category': None,
    },
    'CHED CoScho': {
        'course_category': 'Coconut-Related',
        'course_reason': 'Enrolled in Coconut-Related course',
        'category': 'Coconut-Related',
    },
    'CHED SIDA': {
        'course_category': 'Sugarcane-Related',
        'income_max': 200000,
        'course_reason': 'Enrolled in Sugarcane-Related course',
        'income_reason': 'Family income below ₱200,000 threshold',
        'category': 'Sugarcane-Related',
    },
    'BRO-ED ISU Cauayan': {
        'region': 'Region II',
        'region_reason': 'Located in Region II (Cagayan Valley)',
    },
    'DOST Undergraduate Scholarship': {
        'gwa_min': 85.0,
        'is_stem': True,
        'is_dost_field': True,
        'gwa_reason': 'GWA of 85%+ meets DOST requirement',
        'stem_reason': 'STEM strand background',
        'field_reason': 'Enrolled in DOST priority field (Engineering/Science/IT/Health)',
    },
    'CHED TES': {
        'income_max': 290000,
        'has_vulnerability': True,
        'income_reason': 'Family income below ₱290,000 threshold',
        'vuln_reason': 'Social vulnerability status recognized',
    },
    'CHED SIKAP': {
        'income_min': 300000,
        'income_max': 430000,
        'income_reason': 'Income qualifies for SIKAP program',
    },
    'ACEF-GIAHEP': {
        'gwa_min': 82.0,
        'gwa_max': 95.0,
        'is_agriculture': True,
        'gwa_reason': 'GWA between 82-95% qualifies',
        'agri_reason': 'Enrolled in Agriculture-related course',
    },
    'CHED TDP': {
        'category': None,
    },
    'No Scholarship Recommended': {
        'income_min': 430000,
        'no_reason': 'Income exceeds scholarship thresholds',
    },
}

DOST_FIELDS = {'Engineering', 'Health Sciences', 'Science', 'IT/Computing', 'Agriculture'}


def generate_match_reasons(profile, scholarship_name, confidence):
    """Generate reasons why a scholarship matches a student's profile based on ML features."""
    reasons = []
    rules = SCHOLARSHIP_RULES.get(scholarship_name, {})
    
    gwa = profile.gwa_percentage if hasattr(profile, 'gwa_percentage') else 0
    if not gwa:
        gwa = 100 - (profile.gwa_numeric_1to5 - 1) * 7.5 if profile.gwa_numeric_1to5 else 0
    
    income = profile.family_annual_income_php or 0
    course_category = profile.course_category or ''
    
    # GWA-based matching
    gwa_min = rules.get('gwa_min')
    gwa_max = rules.get('gwa_max', 100)
    if gwa_min and gwa >= gwa_min and gwa <= gwa_max:
        reasons.append(f"✓|{rules.get('gwa_reason', f'GWA of {gwa:.1f}% meets requirement')}")
    elif gwa_min and gwa < gwa_min:
        diff = gwa_min - gwa
        if diff < 5:
            reasons.append(f"~|GWA close to {gwa_min}% requirement")
    
    # Income-based matching
    income_max = rules.get('income_max')
    income_min = rules.get('income_min')
    if income_max and income <= income_max:
        reasons.append(f"✓|{rules.get('income_reason', 'Financial need qualifies')}")
    elif income_min and income >= income_min and income <= rules.get('income_max', float('inf')):
        reasons.append(f"✓|{rules.get('income_reason', 'Income level qualifies')}")
    elif income_max and income > income_max:
        reasons.append(f"~|Income above ₱{income_max:,} threshold")
    
    # Course category matching
    required_category = rules.get('course_category')
    if required_category and course_category == required_category:
        reasons.append(f"✓|{rules.get('course_reason', f'Enrolled in {required_category} course')}")
    elif required_category and course_category != required_category:
        reasons.append(f"~|Not enrolled in {required_category} course")
    
    # DOST field matching
    if rules.get('is_dost_field') and course_category in DOST_FIELDS:
        reasons.append(f"✓|{rules.get('field_reason', 'DOST priority field')}")
    elif rules.get('is_dost_field') and course_category not in DOST_FIELDS:
        reasons.append(f"~|Course not in DOST priority fields")
    
    # STEM strand matching
    if rules.get('is_stem'):
        shs_strand = getattr(profile, 'shs_strand', '') or 'STEM'
        if shs_strand == 'STEM':
            reasons.append(f"✓|{rules.get('stem_reason', 'STEM strand background')}")
        else:
            reasons.append(f"~|Non-STEM strand background")
    
    # Agriculture field matching
    if rules.get('is_agriculture'):
        agri_categories = {'Agriculture', 'Coconut-Related', 'Sugarcane-Related'}
        if course_category in agri_categories:
            reasons.append(f"✓|{rules.get('agri_reason', 'Agriculture-related course')}")
        else:
            reasons.append(f"~|Not in Agriculture field")
    
    # Region matching
    required_region = rules.get('region')
    if required_region:
        student_region = getattr(profile, 'region', '') or 'Region II'
        if required_region in student_region or student_region in required_region:
            reasons.append(f"✓|{rules.get('region_reason', f'Located in {required_region}')}")
        else:
            reasons.append(f"~|Not from {required_region}")
    
    # Vulnerability status matching
    if rules.get('has_vulnerability'):
        vuln_score = (
            (1 if profile.is_4ps_beneficiary else 0) +
            (1 if profile.is_solo_parent_dependent else 0) +
            (1 if profile.is_pwd else 0) +
            (1 if profile.is_indigenous_people else 0) +
            (1 if profile.is_ofw_dependent else 0)
        )
        if vuln_score > 0:
            reasons.append(f"✓|{rules.get('vuln_reason', 'Social vulnerability recognized')}")
        else:
            reasons.append(f"~|No vulnerability status claimed")
    
    # Add specific vulnerability flags if applicable
    if profile.is_4ps_beneficiary and not rules.get('has_vulnerability'):
        reasons.append("✓|4Ps beneficiary status")
    if profile.is_pwd:
        reasons.append("✓|PWD status recognized")
    if profile.is_indigenous_people:
        reasons.append("✓|Indigenous People member")
    if profile.is_solo_parent_dependent:
        reasons.append("✓|Solo parent dependent")
    if profile.is_ofw_dependent:
        reasons.append("✓|OFW dependent status")
    
    # If still no reasons, add generic ones based on confidence
    if len([r for r in reasons if r.startswith('✓')]) == 0:
        if confidence >= 70:
            reasons.append("✓|Profile matches scholarship criteria")
        elif confidence >= 50:
            reasons.append("~|Partially matches requirements")
        else:
            reasons.append("~|Limited match with requirements")
    
    # Limit to 5 most relevant reasons (prioritize ✓ over ~)
    positive = [r for r in reasons if r.startswith('✓')]
    partial = [r for r in reasons if r.startswith('~')]
    final_reasons = positive[:4] + partial[:max(0, 4-len(positive))]
    
    return '\n'.join(final_reasons[:5])


@login_required(login_url='home')
def student_form_view(request):
    if request.method == 'POST':
        form = StudentProfileForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data
            try:
                with transaction.atomic():
                    profile = StudentProfile.objects.create(
                        user=request.user,
                        first_name=cd['first_name'],
                        last_name=cd['last_name'],
                        age=cd['age'],
                        sex=cd['sex'],
                        civil_status=cd['civil_status'],
                        year_level=cd['year_level'],
                        gwa_percentage=cd['gwa_percentage'],
                        gwa_numeric_1to5=cd['gwa_numeric_1to5'],
                        course=cd['course'],
                        course_category=cd['course_category'],
                        income_range=cd['income_range'],
                        family_annual_income_php=cd['family_annual_income_php'],
                        parents_occupation=cd['parents_occupation'],
                        is_solo_parent_dependent=cd['is_solo_parent_dependent'],
                        is_pwd=cd['is_pwd'],
                        is_indigenous_people=cd['is_indigenous_people'],
                        is_4ps_beneficiary=cd['is_4ps_beneficiary'],
                        is_ofw_dependent=cd['is_ofw_dependent'],
                        has_existing_scholarship=cd['has_existing_scholarship'],
                    )

                    student_dict = form.to_student_dict()
                    results = recommend_scholarship(student_dict, top_n=3)

                    for rec in results:
                        scholarship_obj, _ = Scholarship.objects.get_or_create(
                            name=rec['scholarship'],
                        )
                        match_reasons = generate_match_reasons(profile, rec['scholarship'], rec['confidence'])
                        Recommendation.objects.create(
                            student_profile=profile,
                            scholarship=scholarship_obj,
                            rank=rec['rank'],
                            confidence_score=rec['confidence'],
                            match_reasons=match_reasons,
                        )
            except FileNotFoundError:
                messages.error(
                    request,
                    'Scholarship recommendation models are unavailable. Please try again later.',
                )
                return render(request, 'scholarships/student_form.html', {'form': form})

            return redirect('scholarships:results', profile_id=profile.pk)
    else:
        form = StudentProfileForm()

    return render(request, 'scholarships/student_form.html', {'form': form})


@login_required(login_url='home')
def results_view(request, profile_id):
    profile = get_object_or_404(StudentProfile, pk=profile_id, user=request.user)
    recommendations = profile.recommendations.select_related('scholarship').order_by('rank')

    return render(request, 'scholarships/results.html', {
        'profile': profile,
        'recommendations': recommendations,
    })


@login_required(login_url='home')
def scholarship_detail_view(request, profile_id, recommendation_id):
    profile = get_object_or_404(StudentProfile, pk=profile_id, user=request.user)
    recommendation = get_object_or_404(
        Recommendation, pk=recommendation_id, student_profile=profile
    )
    scholarship = recommendation.scholarship

    return render(request, 'scholarships/scholarship_detail.html', {
        'profile': profile,
        'recommendation': recommendation,
        'scholarship': scholarship,
    })


@login_required(login_url='home')
def history_view(request):
    profiles = StudentProfile.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'scholarships/history.html', {
        'profiles': profiles,
    })
