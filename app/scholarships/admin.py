from django.contrib import admin

from .models import Scholarship, StudentProfile, Recommendation


class RecommendationInline(admin.TabularInline):
    model = Recommendation
    readonly_fields = ['scholarship', 'rank', 'confidence_score', 'created_at']
    extra = 0


@admin.register(Scholarship)
class ScholarshipAdmin(admin.ModelAdmin):
    list_display = ['name', 'is_active']
    list_filter = ['is_active']
    search_fields = ['name']


@admin.register(StudentProfile)
class StudentProfileAdmin(admin.ModelAdmin):
    list_display = [
        '__str__', 'course_category', 'region', 'gwa_percentage',
        'family_annual_income_php', 'created_at',
    ]
    list_filter = ['course_category', 'region', 'shs_strand', 'created_at']
    search_fields = ['first_name', 'last_name']
    inlines = [RecommendationInline]
    readonly_fields = ['created_at']


@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['student_profile', 'scholarship', 'rank', 'confidence_score', 'created_at']
    list_filter = ['scholarship', 'rank']
