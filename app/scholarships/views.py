from django.contrib.auth.decorators import login_required
from django.shortcuts import render, redirect, get_object_or_404

from .forms import StudentProfileForm
from .models import StudentProfile, Scholarship, Recommendation
from .ml.engine import recommend_scholarship


@login_required(login_url='home')
def student_form_view(request):
    if request.method == 'POST':
        form = StudentProfileForm(request.POST)
        if form.is_valid():
            cd = form.cleaned_data

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
                Recommendation.objects.create(
                    student_profile=profile,
                    scholarship=scholarship_obj,
                    rank=rec['rank'],
                    confidence_score=rec['confidence'],
                )

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
def history_view(request):
    profiles = StudentProfile.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'scholarships/history.html', {
        'profiles': profiles,
    })
