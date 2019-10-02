class Hacktoberfest

  @@participants_list = [
    "RyanJamesCaldwell"
  ]

  def initialize
    participants
  end

  def participants
    @@participants_list.each do |participant|
      puts "#{participant} contributed to this repo as a part of Hacktoberfest!"
    end
  end
end
