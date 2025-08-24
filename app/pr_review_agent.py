import os
import sys
from openai import OpenAI
from github import Github

MAX_INPUT_LENGTH = 30000  # Maximum length (characters) of the input diff
MAX_OUTPUT_TOKENS = 2000
TEMPERATURE = 0.3
MODEL = "gpt-4o-mini"
FORCE_REVIEW = False
LINES_CHANGE_THRESHOLD = 10

def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = ["OPENAI_API_KEY", "GIT_TOKEN", "GIT_REPOSITORY", "PR_NUMBER"]
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)

def get_pr_diff(pr):
    """Get the diff for a pull request."""
    files = list(pr.get_files())  # Convert PaginatedList to regular list
    diff = ""
    for idx, file in enumerate(files):
        diff += f"Filename: {file.filename}\n"
        diff += f"Status: {file.status}\n"  # 'added', 'removed', 'modified'
        diff += f"Patch (diff):\n{file.patch}\n"
        if idx < len(files) - 1:
            diff += "----" * 10 + "\n"
    print(f"🔍 Diff length: {len(diff)} characters")
    if len(diff) > MAX_INPUT_LENGTH:
        print(f"🔍 Diff truncated to {MAX_INPUT_LENGTH} characters")

    return diff[:MAX_INPUT_LENGTH]  # truncate if needed

def generate_review(diff, client, existing_review=None):
    """Generate review using OpenAI API with error handling."""
    try:
        # Prepare the system message
        system_content = """You are an expert GitHub code reviewer.
                          You are also a senior software & AI engineer with 10+ years of experience in machine learning, Python, and PyTorch.
                          Provide constructive feedback on the PR diff.
                          Deliver a compact, information-dense response and Focus on key points only.
                          Focus on code quality, best practices, potential bugs, and improvement suggestions.
                          If you are satisfied with the PR, you can just say "🤖 I'm satisfied with the PR".
                          NOTE: Your response must be in Korean, but retain technical terms (e.g., PR, Github, CI, Docker, etc.) and the names of variables, functions, etc., in English.
                          Here is the enhanced format for the review:
                          1. **Summary of the PR**: Provide a brief overview of the purpose and scope of the pull request.
                          2. **What's Changed?**: Highlight the key changes made in the codebase.
                          3. **Improvement Suggestions**: Offer constructive feedback on how the code can be improved.
                          4. **Code Quality**: Assess the readability, maintainability, and efficiency of the code.
                          5. **Best Practices**: Evaluate adherence to coding standards and industry best practices.
                          6. **Potential Bugs**: Identify any potential issues or bugs in the code.
                        """
                        #   7. **Testing and Validation**: Comment on the adequacy of tests and validation methods used.
                        #   8. **Documentation**: Review the quality and completeness of documentation provided.
        
        # Prepare the user message
        user_content = f"Please review this PR diff:\n\n{diff}"
        
        # If there's a previous review, add it as context
        if existing_review:
            system_content += "\n\nIMPORTANT: This is a follow-up review. \
                               The diff represents changes since the last review. \
                               Consider previous feedback and provide updated suggestions based on new changes. \
                               Focus on what has changed since the last review. \
                               Avoid repeating content from the previous review. \
                               Identify and indicate which prior suggestions have been implemented and which require further attention. \
                               Use ✅ to denote implemented suggestions and ❌ for those not yet addressed."
            user_content += f"\n\nPrevious review:\n{existing_review.body}"
        
        print(f"🔍 System content length: {len(system_content)} characters")
        print(f"🔍 User content length: {len(user_content)} characters")
        if existing_review:
            print(f"🔍 Existing review length: {len(existing_review.body)} characters")


        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user", 
                    "content": user_content
                }
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE
        )
        print(f"🔍 Prompt tokens: {response.usage.prompt_tokens} tokens")
        print(f"🔍 Completion tokens: {response.usage.completion_tokens} tokens")
        print(f"🔍 Total tokens: {response.usage.total_tokens} tokens")
        print(f"🔍 Completion length: {len(response.choices[0].message.content)} characters")
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Failed to generate review: {e}")
        sys.exit(1)

def check_existing_agent_review(pr):
    """Check if the agent has already reviewed this PR."""
    try:
        reviews = pr.get_reviews()
        if not reviews:
            print("🤖 This is the first review")
            return None
        latest_agent_review = None
        for review in reviews:
            if "🤖" in review.body:
                if latest_agent_review is None or review.submitted_at > latest_agent_review.submitted_at:
                    latest_agent_review = review
        if latest_agent_review:
            print(f"⚠️  Latest agent review found (ID: {latest_agent_review.id})")
        return latest_agent_review
    except Exception as e:
        print(f"❌ Failed to check existing reviews: {e}")
        return None

def get_lines_changed_since_review(pr, last_review):
    """Get the number of lines changed since the last review."""
    try:
        # Get the commit SHA when the last review was submitted
        last_review_commit = last_review.commit_id
        
        # Get the current head commit
        current_commit = pr.head.sha
        
        # If it's the same commit, no changes
        if last_review_commit == current_commit:
            return 0
        
        # Get the diff between the last review commit and current head
        comparison = pr.base.repo.compare(last_review_commit, current_commit)
        
        total_additions = 0
        total_deletions = 0
        
        for file in comparison.files:
            total_additions += file.additions
            total_deletions += file.deletions
        
        total_changes = total_additions + total_deletions
        print(f"📊 Lines changed since last review: +{total_additions} -{total_deletions} = {total_changes} total")
        
        return total_changes
        
    except Exception as e:
        print(f"❌ Failed to get lines changed: {e}")
        return 0

def check_significant_update(pr, existing_review):
    """Check if we should force review due to significant changes."""
    try:
        lines_changed = get_lines_changed_since_review(pr, existing_review)
        
        if lines_changed > LINES_CHANGE_THRESHOLD:
            print(f"🔄 Significant changes detected ({lines_changed} lines > {LINES_CHANGE_THRESHOLD} lines (threshold))")
            print(f"💡 Adding new review due to substantial changes")
            return True
        else:
            print(f"✅ Changes are minimal ({lines_changed} lines ≤ {LINES_CHANGE_THRESHOLD} lines (threshold))")
            return False
            
    except Exception as e:
        print(f"❌ Failed to check change threshold: {e}")
        return False

def should_proceed_with_review(pr, existing_review):
    """Check if we should proceed with a new review based on existing agent reviews and changes."""
    if FORCE_REVIEW:
        print("🔄 Force review enabled - will regenerate even if review exists")
        return True
    
    if existing_review:
        return check_significant_update(pr, existing_review)

    return True

def get_diff_since_prev_review(pr, last_review):
    """Get the diff for commits made after the last review."""
    try:
        # Get the commit SHA when the last review was submitted
        last_review_commit = last_review.commit_id
        
        # Get the current head commit
        current_commit = pr.head.sha
        
        # If it's the same commit, no changes
        if last_review_commit == current_commit:
            return ""
        
        # Get the diff between the last review commit and current head
        comparison = pr.base.repo.compare(last_review_commit, current_commit)
        
        diff = ""
        for idx, file in enumerate(comparison.files):
            diff += f"Filename: {file.filename}\n"
            diff += f"Status: {file.status}\n"  # 'added', 'removed', 'modified'
            diff += f"Patch (diff):\n{file.patch}\n"
            if idx < len(comparison.files) - 1:
                diff += "----" * 10 + "\n"
        
        print(f"🔍 Diff since last review length: {len(diff)} characters")
        if len(diff) > MAX_INPUT_LENGTH:
            print(f"🔍 Diff since last review truncated to {MAX_INPUT_LENGTH} characters")
        
        return diff[:MAX_INPUT_LENGTH]  # truncate if needed
        
    except Exception as e:
        print(f"❌ Failed to get diff since review: {e}")
        return ""

def generate_line_suggestions(diff, client, existing_review=None):
    """Generate line-specific suggestions using OpenAI API."""
    try:
        system_content = """You are an expert code reviewer. Analyze the diff and provide specific line-by-line suggestions.
                          For each suggestion, provide:
                          1. The filename where the change should be made
                          2. The exact line number (or range) where the change should be made
                          3. The specific code change or suggestion
                          4. A brief explanation of why this change is needed
                          
                          Format your response as a JSON array with objects containing:
                          {
                            "file": "<filename>",
                            "line": <line_number>,
                            "suggestion": "<specific code suggestion>",
                            "explanation": "<brief explanation>"
                          }
                          
                          Only provide suggestions for actual issues that need fixing.
                          If no specific line changes are needed, return an empty array [].
                          """
        
        user_content = f"Please analyze this diff and provide line-specific suggestions:\n\n{diff}"
        
        if existing_review:
            user_content += f"\n\nPrevious review context:\n{existing_review.body}"
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1000,
            temperature=0.2
        )
        
        suggestions_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON response
        try:
            import json
            suggestions = json.loads(suggestions_text)
            if isinstance(suggestions, list):
                return suggestions
            else:
                print("⚠️  Unexpected suggestions format, treating as general comment")
                return []
        except json.JSONDecodeError:
            print("⚠️  Could not parse suggestions as JSON, treating as general comment")
            return []
            
    except Exception as e:
        print(f"❌ Failed to generate line suggestions: {e}")
        return []

def post_review_with_suggestions(pr, review_text, suggestions, diff):
    """Post review with both general comments and line-specific suggestions."""
    try:
        # Post the main review comment
        pr.create_review(
            body=f"## 🤖 PR Review Agent Suggestion\n\n{review_text}",
            event="COMMENT"
        )
        
        # Post line-specific suggestions if any
        if suggestions:
            print(f"📝 Posting {len(suggestions)} line-specific suggestions...")
            
            # Get all files in the PR for mapping
            pr_files = {file.filename: file for file in pr.get_files()}
            
            for suggestion in suggestions:
                try:
                    filename = suggestion.get('file')
                    line_num = suggestion.get('line')
                    suggestion_text = suggestion.get('suggestion', '')
                    explanation = suggestion.get('explanation', '')
                    
                    if filename and line_num and suggestion_text:
                        # Check if the file exists in the PR
                        if filename in pr_files:
                            file = pr_files[filename]
                            
                            # Create the comment body
                            comment_body = f"🤖 **Suggestion:**\n```\n{suggestion_text}\n```\n\n**Reason:** {explanation}"
                            
                            # Create a review comment on the specific line
                            pr.create_review_comment(
                                body=comment_body,
                                commit_id=pr.head.sha,
                                path=filename,
                                position=line_num
                            )
                            print(f"✅ Posted suggestion for line {line_num} in {filename}")
                        else:
                            print(f"⚠️  File {filename} not found in PR, skipping suggestion")
                    else:
                        print(f"⚠️  Invalid suggestion format, skipping: {suggestion}")
                        
                except Exception as e:
                    print(f"❌ Failed to post suggestion for {suggestion.get('file', 'unknown')}:{suggestion.get('line', 'unknown')}: {e}")
                    continue
        
        print("✅ Successfully posted review with suggestions.")
        
    except Exception as e:
        print(f"❌ Failed to post review: {e}")
        sys.exit(1)

def main():
    """Main function to run the PR review agent."""
    print("🤖 Starting PR Review Agent...")
    
    # Validate environment variables
    validate_environment()
    
    # Load configuration
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    git_token = os.environ.get("GIT_TOKEN")
    repo_name = os.environ.get("GIT_REPOSITORY")
    pr_number = int(os.environ.get("PR_NUMBER"))

    # Initialize clients
    try:
        gh = Github(git_token)
        print(f"🔐 Authenticated as: {gh.get_user().login}")
        
        repo = gh.get_repo(repo_name)
        print(f"📁 Repository: {repo.full_name}")
        print(f"🔗 Repository URL: {repo.html_url}")
        
        pr = repo.get_pull(pr_number)
        print(f"✅ PR #{pr_number} found successfully")
        print(f"📋 PR Title: {pr.title}")
        print(f"👤 Author: {pr.user.login}")
        print(f"📊 PR State: {pr.state}")
        
        client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        print(f"❌ Failed to initialize clients: {e}")
        sys.exit(1)
    
    print(f"📋 Reviewing PR #{pr_number} in {repo_name}")

    # Check for existing review once
    existing_review = check_existing_agent_review(pr)
    
    if not should_proceed_with_review(pr, existing_review):
        return

    # Get PR diff
    if existing_review:
        print("📋 Getting diff since last review...")
        diff = get_diff_since_prev_review(pr, existing_review)
    else:
        print("📋 Getting full PR diff...")
        diff = get_pr_diff(pr)
        
    if not diff.strip():
        print("⚠️  No changes found in PR diff")
        return
        
    # Generate review
    print("🤖 Generating review...")
    if existing_review:
        print("📋 Including previous review as context...")
    review_text = generate_review(diff, client, existing_review)

    # Generate line-specific suggestions
    print("🤖 Generating line-specific suggestions...")
    suggestions = generate_line_suggestions(diff, client, existing_review)

    # Post review with suggestions
    print("🤖 Posting review with suggestions...")
    post_review_with_suggestions(pr, review_text, suggestions, diff)

if __name__ == "__main__":
    main()
