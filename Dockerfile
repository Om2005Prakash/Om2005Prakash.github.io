FROM ruby:3.1

WORKDIR /srv/jekyll

RUN gem install bundler

COPY Gemfile ./

RUN bundle install

COPY . .

EXPOSE 4000

CMD ["bundle", "exec", "jekyll", "serve", "--host", "0.0.0.0", "--livereload"]