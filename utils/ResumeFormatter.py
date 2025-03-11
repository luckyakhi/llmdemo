from fpdf import FPDF

# Resume text
resume_text = {
    "name": "Akhilesh Singh",
    "contact": "Bengaluru, 562125\n+91-9769132490\nakhi.iitkgp@gmail.com",
    "summary": "Accomplished software developer with 16 years of experience specializing in Java and Microservices architecture, complemented by proficiency in Python development. Proven track record in leading development teams, designing scalable systems, and implementing efficient solutions in complex, data-driven environments. AWS Certified with a strong background in Spark, Spring Boot, and Kubernetes.",
    "experience": [
        {
            "company": "JPMorgan Chase, Bengaluru",
            "role": "Vice President",
            "duration": "January 2021 - Present",
            "projects": [
                {
                    "title": "JPMC Private Banks Party and Account Data Platform",
                    "duration": "June 2022 - Current",
                    "role": "Java + Python Developer",
                    "responsibilities": [
                        "Implemented Spring Boot-based Microservices to support the platform.",
                        "Designed and implemented a Kafka-based distributed architecture to enable asynchronous distribution to various data stores.",
                        "Developed capabilities to check distribution status across multiple data stores.",
                        "Implemented a Python project focused on automating testing workflows, utilizing Python frameworks to generate and validate complex test scenarios.",
                        "Implemented a RAG-based solution to auto-assign incident tickets based on microservice owners, leveraging Large Language Models (LLM) to enhance incident management processes.",
                        "Conducted technical sessions for the team on Large Language Models (LLM) and Agent-based RAG architecture, promoting knowledge sharing and adoption of these technologies within the team."
                    ]
                },
                {
                    "title": "Spark-based Regulatory Reporting Application",
                    "duration": "June 2020 - June 2022",
                    "role": "Scrum Master & Lead Java Developer",
                    "responsibilities": [
                        "Designed and architected a configurable and scalable distributed data processing framework using Spark to generate regulatory reports with critical SLAs.",
                        "Implemented the framework in Java & Spring, enabling the generation of Spark transformations based on data source/target and workflow configuration in JSON.",
                        "Developed Spring Boot-based Microservices for metadata management and data quality controls.",
                        "Led the effort to move Spark jobs to a Kubernetes managed cluster.",
                        "Served as Scrum lead supporting critical regulatory reports."
                    ]
                }
            ]
        },
        {
            "company": "Morgan Stanley, Bengaluru",
            "role": "Consultant (Publicis.Sapient)",
            "duration": "October 2018 - May 2020",
            "projects": [
                {
                    "title": "Java & Spark Data Ingestion Framework for Operations Tech",
                    "duration": "",
                    "role": "Lead Java & Hadoop Developer",
                    "responsibilities": [
                        "Implemented a Java & Spring-based data ingestion framework to hydrate the Hadoop data lake.",
                        "Developed an Angular UI & Spring Microservice-based full-stack application for monitoring ingestion status."
                    ]
                }
            ]
        },
        {
            "company": "Wells Fargo, Bengaluru",
            "role": "Consultant (Publicis.Sapient)",
            "duration": "October 2016 - September 2018",
            "projects": [
                {
                    "title": "Enterprise Data Lake Framework",
                    "duration": "",
                    "role": "Lead Java & Hadoop Developer",
                    "responsibilities": [
                        "Implemented Java+Spring-based Microservices to support Enterprise data lake platform.",
                        "Implemented Real-Time Data Ingestion Microservice leveraging Kafka.",
                        "Enhanced Spark-based Data Quality Microservice by adding the ability to generate Spark jobs based on Business Rules."
                    ]
                }
            ]
        },
        {
            "company": "JPMorgan Chase, Mumbai",
            "role": "Associate",
            "duration": "June 2014 - September 2016",
            "projects": [
                {
                    "title": "Security Lending Middleware Platform",
                    "duration": "",
                    "role": "Java & Spring Developer",
                    "responsibilities": [
                        "Developed a Java & Spring-based middleware service platform to support Security lending transactions.",
                        "Leveraged Java concurrency framework to support highly concurrent low-latency trading applications."
                    ]
                }
            ]
        },
        {
            "company": "Deloitte Consulting, Mumbai",
            "role": "Systems Engineer",
            "duration": "June 2008 - June 2014",
            "projects": [
                {
                    "title": "IBM Global Move Project & US Tennessee Government Welfare Project",
                    "duration": "",
                    "role": "Java & Spring Developer",
                    "responsibilities": [
                        "Implemented critical components of IBM Global Move solution using Java & IBM Websphere portal technology.",
                        "Implemented Java & Struts-based framework to support Child care solutions of the State of Tennessee, US."
                    ]
                }
            ]
        }
    ],
    "skills": [
        "Core Java",
        "Spring Boot",
        "Microservices",
        "AWS",
        "Spark",
        "Kafka",
        "RDBMS (Oracle, MySQL)",
        "NoSQL (MongoDB)",
        "Kubernetes",
        "Docker",
        "Angular",
        "Python",
        "Large Language Models (LLM)",
        "Retrieval-Augmented Generation (RAG) Architecture"
    ],
    "education": "Bachelor of Technology in Computer Science, IIT Kharagpur, 2008",
    "certifications": [
        "AWS Certified Solutions Architect",
        "Large Language Models (LLM) Certification, DataCamp"
    ]
}

# Create a PDF object
pdf = FPDF()

# Add a page
pdf.add_page()

# Set font and size for name
pdf.set_font("Arial", style='B', size=16)

# Add name centered
pdf.cell(0, 10, txt=resume_text['name'], ln=True, align='C')

# Set font and size for contact info
pdf.set_font("Arial", size=10)

# Add contact info centered
for line in resume_text['contact'].split('\n'):
    pdf.cell(0, 10, txt=line, ln=True, align='C')

# Set font and size for summary header
pdf.set_font("Arial", style='B', size=14)
pdf.cell(0, 10, txt="Summary", ln=True)

# Set font and size for summary text
pdf.set_font("Arial", size=10)
pdf.multi_cell(0, 10, txt=resume_text['summary'])

# Set font and size for experience
pdf.set_font("Arial", style='B', size=14)
pdf.cell(0, 10, txt="Work Experience", ln=True)

for experience in resume_text['experience']:
    # Company and role
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, txt=f"{experience['company']}\n{experience['role']}\n{experience['duration']}", ln=True)

    for project in experience['projects']:
        # Project title
        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(0, 10, txt=f"Project: {project['title']}\nDuration: {project['duration']}\nRole: {project['role']}",
                 ln=True)

        # Responsibilities
        pdf.set_font("Arial", size=10)
        for responsibility in project['responsibilities']:
            pdf.cell(0, 10, txt=f"- {responsibility}", ln=True)

# Set font and size for skills
pdf.set_font("Arial", style='B', size=14)
pdf.cell(0, 10, txt="Skills", ln=True)
pdf.set_font("Arial", size=10)
for skill in resume_text['skills']:
    pdf.cell(0, 10, txt=f"- {skill}", ln=True)

# Set font and size for education
pdf.set_font("Arial", style='B', size=14)
pdf.cell(0, 10, txt="Education", ln=True)
pdf.set_font("Arial", size=10)
pdf.cell(0, 10, txt=resume_text['education'], ln=True)

# Set font and size for certifications
pdf.set_font("Arial", style='B', size=14)
pdf.cell(0, 10, txt="Certifications", ln=True)
pdf.set_font("Arial", size=10)
for certification in resume_text['certifications']:
    pdf.cell(0, 10, txt=f"- {certification}", ln=True)

# Save the PDF with filename
pdf.output("AkhileshSinghResume.pdf")

print("PDF generated successfully.")
