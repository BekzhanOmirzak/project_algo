package interview

abstract class Person {
    abstract fun walk()
}

class Man : Person() {
    override fun walk() {
        println("Man is walking")
    }
}

class Woman : Person() {
    override fun walk() {
        println("Women is walking")
    }
}

//DRY
//LiskovSubstitution
fun makeWalk(person: Person){
    person.walk()
}
