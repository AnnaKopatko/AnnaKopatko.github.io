---
layout: default
title: "Anna Kopatko"
subtitle: "Perception Engineer | Deep Learning | Physics Enthusiast"
---

<div style="margin-left: 120px; margin-right: 50px;">

<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 40px;">

<!-- Left side: text -->
<div style="flex: 1; min-width: 300px;">

<h2>Perception Engineer | Deep Learning | Physics Enthusiast</h2>

<p>I'm Anna Kopatko, a deep learning engineer based in Berlin, Germany.  
I hold a bachelor's degree in <b>Theoretical Physics</b>, where I developed a strong foundation in mathematical modeling and complex systems.</p>

<p>Currently, I work in <b>perception engineering</b>, where I <b>research, implement, and deploy</b> diverse <b>neural network models</b> for real-world applications — including object detection, segmentation, and sensor fusion.</p>

<p>I'm passionate about combining <b>physics and neural networks</b> to build <b>robust, explainable AI systems</b> that can transform scientific discovery and technology.</p>

<p>Feel free to explore my projects, blog posts, and get in touch!</p>

</div>

<!-- Right side: photo -->
<div style="flex: 0 0 350px; text-align: center;">

<img src="/assets/anna.jpg" alt="Anna Kopatko" style="width: 320px; height: 320px; object-fit: cover; border-radius: 100%; margin-bottom: 20px;">

<p><b>Berlin, Germany</b></p>

</div>

</div>

</div> <!-- end margin wrapper -->

---

# Latest Posts

<div style="display: flex; flex-direction: column; gap: 20px; margin-top: 20px;">

{% for post in site.posts %}
<div style="padding: 20px; border: 1px solid #333; border-radius: 10px; background-color: #000000;">
    <h3><a href="{{ post.url }}" style="color: #ffcc99; text-decoration: none;">{{ post.title }}</a></h3>
    <p style="color: #ddd;">{{ post.date | date: "%B %d, %Y" }}</p>
    <p>{{ post.excerpt | strip_html | truncatewords: 30 }}</p>
    <a href="{{ post.url }}" style="color: #ffcc99;">Read more →</a>
  </div>
{% endfor %}

</div>
