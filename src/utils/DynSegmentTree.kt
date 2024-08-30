class DynamicSegmentTree(val size: Int) {

    var root = Pointer(0)

    fun add(value: Int) {
        updateHelper(root, value, 0, 0, size)
    }

    private fun updateHelper(root: Pointer, value: Int, at: Int, l: Int, r: Int) {
        if (value !in l..r)
            return
        else if (l == r) {
            root.freq++
            return
        }
        root.freq++
        val mid = (l + r) / 2
        root.init(at)
        updateHelper(root.left!!, value, at * 2 + 1, l, mid)
        updateHelper(root.right!!, value, at * 2 + 2, mid + 1, r)
    }

    fun dfs() {
        dfsHelper(root, 0, size)
    }

    private fun dfsHelper(root: Pointer?, l: Int, r: Int) {
        if (root == null) return
        val mid = (l + r) / 2
        println("Node : ${root.at} Freq : ${root.freq} L : $l  R : $r")
        dfsHelper(root.left, l, mid)
        dfsHelper(root.right, mid + 1, r)
    }

    fun findCountInRange(l: Int, r: Int): Int {
        return rangeHelper(root, 0, size, l, r)
    }

    private fun rangeHelper(at: Pointer?, l: Int, r: Int, qL: Int, qR: Int): Int {
        if (at == null)
            return 0
        if (r < qL || qR < l)
            return 0
        if (qL <= l && r <= qR)
            return at.freq
        val mid = (l + r) / 2
        val left = rangeHelper(at.left, l, mid, qL, qR)
        val right = rangeHelper(at.right, mid + 1, r, qL, qR)
        return left + right
    }

}

class Pointer(var at: Int) {
    var freq: Int = 0
    var left: Pointer? = null
    var right: Pointer? = null

    fun init(at: Int) {
        if (left == null) {
            left = Pointer(at * 2 + 1)
            right = Pointer(at * 2 + 2)
        }
    }

}











