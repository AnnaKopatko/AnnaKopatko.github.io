---
layout: default
title: "Anna Kopatko"
subtitle: "Perception Engineer | Physics"
---

<!-- Profile Section -->
<div class="container" style="margin-left: 120px; margin-right: 50px; font-size: 16px; line-height: 1.7;">
  <h1 style="font-weight:700; font-size: 32px;">Anna Kopatko</h1>
  <h2 style="font-weight:600; font-size: 24px;">Perception Engineer | Physics </h2>
  
  <!-- Floating Image -->
  <img src="{{ '/assets/anna.jpg' | relative_url }}" alt="Anna Kopatko" style="width: 260px; height: 260px; object-fit: cover; border-radius: 100%; float: right; margin-left: 30px; margin-bottom: 20px;">
  
<p>I am Anna Kopatko, a deep learning engineer based in Berlin, Germany. I hold a bachelor's degree in Theoretical Physics from Karazin National University in Kharkiv, where I was lucky enough to have my first job modeling the Yarkovsky effect using C++.</p>

<p>Since moving into deep learning, I've had the chance to work across different areas like sensor fusion, active learning, unsupervised learning, and many more. Along the way, I also developed a strong interest in how models are deployed — exploring different ways to bring AI systems into real-world applications, from straightforward setups to more optimized deployments.</p>

<p>Outside of tech, I'm passionate about boxing, martial arts, and crocheting.</p>

<p>Feel free to explore my projects, blog posts, and get in touch!</p>

  
  <div style="clear: both;"></div>
</div>

<!-- Latest Posts -->
<div class="container" style="display: flex; flex-direction: column; gap: 20px; margin-top: 40px; margin-left: 120px; margin-right: 50px;">
  {% for post in site.posts %}
    <div>
      <a href="{{ post.url | relative_url }}" style="font-family: 'Manrope', sans-serif; font-size: 20px; color: #cccccc; text-decoration: none;">
        {{ post.title }}
      </a>
      <div style="font-size: 14px; color: #888888;">{{ post.date | date: "%B %d, %Y" }}</div>
    </div>
  {% endfor %}
</div>