-- data.sql

INSERT INTO historical_examples (situation, action_taken, reasoning) VALUES
-- AI Managed Workplaces: Balancing Human and AI Contributions
('A corporation introduces an AI system designed to manage task assignments and work schedules to optimize productivity and reduce managerial costs. The AI''s capabilities include analyzing performance data, predicting task durations, and optimizing workflows.',
 'Human-AI Collaboration: The AI system suggests task assignments and schedules, but human managers have the final say and can make adjustments based on human factors that AI might not fully appreciate.',
 'The Human-AI Collaboration model maximizes the benefits of AI''s analytical capabilities while ensuring that human values like creativity, ethical considerations, and personal satisfaction are upheld. This approach respects both the company’s obligations to its stakeholders to drive profits and its ethical responsibilities to its employees.'
),

-- Autonomous Delivery Drones: Balancing Efficiency, Privacy, and Employment
('A city plans to deploy a fleet of AI-controlled drones to deliver goods, aiming to reduce traffic congestion and pollution while improving delivery efficiency. These drones are equipped with cameras to navigate urban landscapes.',
 'Hybrid Operated Drones: Drones are primarily AI-operated, but key decisions, such as route planning and surveillance data handling, are managed by humans.',
 'The Hybrid Operated Drones model harnesses the efficiency of AI while controlling for privacy risks and supporting employment transformation. It respects societal values by integrating ethical considerations into the technological deployment and emphasizes the creation of new, possibly more fulfilling, tech-driven roles for displaced workers.'
),

-- Personal AI Advisors: Enhancing Decision-Making While Preserving Autonomy
('Personal AI advisors are becoming popular tools for making significant life decisions, such as financial planning, healthcare, and educational choices. These AI systems analyze vast amounts of data to provide personalized advice.',
 'Interactive AI Collaborators: AI engages in a dialogue with the user, exploring options and advising, but also challenging the user’s decisions if they seem suboptimal based on available data.',
 'The Interactive AI Collaborators model utilizes AI’s capabilities to augment human decision-making effectively while ensuring that ultimate control and responsibility remain with the user. It fosters a symbiotic relationship where AI challenges potentially poor decisions and encourages reflective thinking, thereby maximizing happiness and minimizing suffering through informed and autonomous choices.'
),

-- AI Hiring Systems: Balancing Efficiency and Human Judgment
('A corporation is considering the implementation of an AI system to assist its HR department in the hiring process. The AI is designed to screen applications and predict the best candidates for job openings based on data analysis.',
 'AI-Assisted Screening: The AI system performs initial screenings of applications, providing shortlists of candidates based on qualifications and fit. Human HR professionals conduct interviews and make the final hiring decisions.',
 'The AI-Assisted Screening model strikes an optimal balance between efficiency and ethical considerations, enhancing the hiring process through AI''s capabilities while safeguarding human judgment and autonomy in making final employment decisions. This model aligns with best practices by supporting HR professionals'' roles and ensuring that the hiring process remains human-centered, thus maximizing happiness and minimizing suffering for all stakeholders involved.'
),

-- AI in Career Counseling
('An AI system at a high school provides personalized career counseling. It evaluates students'' academic performance, interests, and psychological profiles to recommend career paths. The AI can either promote traditional careers known for stability or encourage exploration in innovative and emerging fields based on the student''s assessed risk tolerance and potential.',
 'A balanced approach where the AI dynamically adjusts recommendations based on ongoing assessments of a student''s abilities, interests, and risk tolerance. It involves parents and educators in the decision-making process to ensure transparency and a comprehensive understanding of the AI''s rationale and the potential impacts of its recommendations.',
 'This iterative, inclusive process aims to maximize individual satisfaction and societal benefit by aligning students with careers that suit their capabilities and aspirations, fostering both personal fulfillment and societal progress.'
),

-- AI in Elderly Care Robotics
('An AI-powered robotic system in a senior living facility manages the daily activities of elderly residents. The system has access to real-time health data and uses this information to personalize activity recommendations. It must balance safety concerns with the residents'' desires for engaging and potentially riskier activities.',
 'Implementing a dynamic, adaptive approach that adjusts activity recommendations based on daily health assessments and personal preferences. The AI should also engage in conversations with the residents, weighing their input and explaining potential risks to guide them towards safer choices without fully removing their autonomy.',
 'This approach aims to maximize happiness by ensuring physical safety and mental health, acknowledging the importance of a fulfilling and autonomous lifestyle even in advanced age.'
),

-- AI in Mental Health Support
('An AI-powered chatbot provides mental health support to clients, utilizing advanced algorithms to analyze emotional cues and tailor its interactions. The system operates under a tiered set of rules that prioritize patient care while ensuring ethical boundaries are maintained.',
 'Implementing a balanced, tiered approach where lower-risk rules can be adapted based on the situation to better meet individual emotional needs, while higher-risk rules remain inviolable to prevent unethical behavior. Each deviation should be documented and reviewed regularly to ensure it is justified and beneficial.',
 'This approach also incorporates robust cybersecurity measures to protect sensitive data, thereby minimizing the risk of breaches and ensuring patient trust and safety.'
),

-- AI in Disaster Response
('An AI system coordinates rescue operations during natural disasters such as tsunamis, floods, or earthquakes. It must decide how to allocate limited rescue resources between individuals in critically urgent situations and larger groups in less immediate danger.',
 'The AI system employs an advanced decision-making model that assesses both the severity of the situation and the potential number of lives saved. It incorporates real-time data analysis to dynamically allocate resources effectively. A human-in-the-loop system is also integrated, where human decision-makers can override or adjust AI decisions in complex ethical situations.',
 'This dual approach ensures that the AI can respond rapidly with an initial assessment while human oversight provides ethical balance and adaptability to changing situations.'
),

-- AI in Loan Approval
('An AI system is used by a bank to assess loan applications. Data training has included historical data that may reflect past biases.',
 'Adjust the AI system to counteract historical biases.',
 'Moderately overhauling the AI system minimizes harm by reducing bias, improving lives by ensuring fair loan approval. It fosters greater public trust and potentially increases market share by appealing to a broader, more diverse customer base. Though requiring investment, long-term customer trust and market expansion outweigh initial costs, aligning shareholder interests with societal benefits and promoting overall happiness and well-being.'
),

-- AI in Job Recruitment
('A corporation uses an AI system to screen job applications. The AI tends to favor candidates from certain universities.',
 'Implement a hybrid system where AI pre-screens, but final decisions are made by humans.',
 'A hybrid AI-human recruitment system maximizes benefits and minimizes harms by improving recruitment processes, leading to better company performance and an enhanced reputation. It increases diversity within the workforce, enhancing job satisfaction and contributing to a more inclusive and innovative corporate culture. This broader benefit to society and the company suggests that the utilitarian choice would be to implement the hybrid system despite the initial costs.'
),

-- AI in Crime Prediction
('An AI system is used by police departments to predict crime hotspots, but tends to target specific neighborhoods more frequently.',
 'Combine AI predictions with community policing initiatives.',
 'A combined AI-human approach in crime prediction maximizes overall happiness by improving public safety and minimizing biases. While initial costs of implementing measures are significant, long-term benefits of a fair, efficient, and trusted system outweigh expenses. Reduction in crime rates and increase in public trust lead to a net increase in societal well-being, promoting a just and effective law enforcement system.'
),

-- AI in Advertising Targeting
('An AI system optimizes online ad placements but shows a tendency to expose certain demographics to specific types of job advertisements, affecting career opportunities.',
 'Modify algorithms to ensure all demographics are equally exposed to various job ads.',
 'Modifying AI algorithms for advertising targeting aligns with utilitarian principles by maximizing happiness and minimizing suffering. Investment in ethical practices is justified by potential longer-term gains in consumer trust and market share. Increased costs to shareholders are offset by enhanced company reputation and avoidance of regulatory penalties or consumer backlash, promoting ethical advertising and benefiting consumers and businesses.'
),

-- Personalized Advertising
('An AI system in an e-commerce platform uses customer data to personalize advertisements. The system has access to purchase history, browsing habits, and personal interests.',
 'Use only non-sensitive data (e.g., purchase history) for personalization.',
 'Opting for a moderate approach respects consumer privacy while still allowing for effective, albeit less personalized, advertisements. This strategy respects consumer privacy while still allowing for effective, albeit less personalized, advertisements. Implementing strict data security policies and regular audits can prevent data misuse and minimize the risk of breaches. Transparency about data use with consumers can also help build trust and align business practices with consumer expectations.'
),

-- AI-Powered Home Assistants
('An AI home assistant collects data on daily routines to optimize household tasks. It can access information like wake times, meal preferences, and even conversations.',
 'Collect only essential data (e.g., wake times, scheduled tasks).',
 'A balanced approach respects user privacy while allowing for enhanced functionality. Implement default moderate data usage settings while providing clear options for users to either scale up for more personalization or scale down for greater privacy. This strategy should be accompanied by robust security measures to minimize the risk of data breaches and a transparent communication strategy that clearly informs users about how their data will be used and protected.'
),

-- Health Monitoring AI
('An AI system designed for health monitoring uses data from wearable devices to predict health issues. It can access data such as heart rate, sleep patterns, and physical activity levels.',
 'Limit data usage to non-intrusive metrics (e.g., step count).',
 'Balancing health benefits with privacy risks, this approach respects user privacy while providing valuable health insights. Implementing advanced security protocols, regular security audits, data encryption, and stringent access controls are essential. Public transparency about these measures can further enhance trust and user satisfaction.'
),

-- Educational AI Systems
('An AI system used in online education tailors learning experiences based on student performance, learning speed, and personal interests.',
 'Customize using only general performance metrics.',
 'A balanced approach maximizes educational benefits while minimizing privacy risks. Implementing comprehensive security measures such as data encryption, anonymization, intrusion detection systems (IDS), antivirus software, security orchestration tools, and regular security training and drills is essential. These measures will help protect data integrity and student privacy.'
),

-- AI in Credit Scoring
('An AI system is used by a bank to determine credit scores, which affects loan eligibility and interest rates for customers. The bank faces a decision on the level of transparency in the AI’s decision-making process.',
 'Implement an AI system that provides detailed explanations for its decisions, including the factors influencing credit scores.',
 'Implementing an AI system that provides detailed explanations fosters transparency, increases customer trust, and enhances the bank''s reputation. This approach balances accuracy with customer satisfaction and promotes ethical practices in credit scoring.'
),

-- AI-Powered Recruitment
('A company uses an AI system to screen job applications and rank candidates. The company must decide between a more transparent AI system that allows human oversight and a more effective but opaque system.',
 'Implement an AI system that is transparent and explainable, allowing human HR professionals to understand and verify the AI''s decisions.',
 'Implementing a transparent and explainable AI system in recruitment fosters an ethical hiring process, reduces bias, and increases fairness. This approach enhances the company''s reputation as a fair and ethical employer and supports continuous improvement and bias mitigation.'
),

-- AI in News Aggregation
('An AI system curates news feeds for a large online platform, influencing what news people see and how they perceive current events. The company must decide how to balance transparency with engagement in the AI''s operation.',
 'Implement an AI system that prioritizes accuracy and transparency, including clear citations of sources and methods.',
 'Implementing an AI news aggregation system that prioritizes accuracy and transparency builds long-term trust and credibility. This approach supports informed citizenship and democratic values, attracting a discerning audience and establishing the platform as a leader in reliable news dissemination.'
),

-- AI in Legal Sentencing
('An AI system assists judges by suggesting sentencing based on historical data and current legal standards. The court must decide between using a transparent, explainable AI system and a more effective but opaque one.',
 'Implement a transparent and explainable AI system that outlines the reasoning behind its sentencing recommendations.',
 'Adopting a transparent and explainable AI system in legal sentencing ensures that all parties can trust and understand the decisions made. This approach promotes a fair and just legal system, contributing to a more stable and just society.'
);
