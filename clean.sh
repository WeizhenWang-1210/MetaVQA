for d in [0-9]*_[0-9]*_[0-9]*; do
    if [ -d "$d" ]; then
        rm -rf "$d"
    fi
done