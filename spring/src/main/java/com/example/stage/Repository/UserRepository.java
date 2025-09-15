package com.example.stage.Repository;


import com.example.stage.Model.User;
import org.springframework.data.mongodb.repository.MongoRepository;

public interface UserRepository extends MongoRepository<User, String> {
    User findByUsername(String username);
     User findByEmail(String email);
}
